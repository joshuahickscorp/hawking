# Hawking IDE — Vision and Constitution

> **Chapter 00 of 13 — Read this first.**
> This document is the founding specification of the Hawking IDE (HIDE). It states what the product is, why it exists, the principles that govern every architecture decision, and the single gate measurement on which the entire build depends. Every subsequent chapter assumes fluency with the material here.

---

## §0.1 — Product Vision

### The Problem

Cloud AI coding assistants are extraordinary and getting better. They are also structurally compromised for a large and growing class of developers: those who cannot or will not ship their source code to a third-party API, those who are tired of metered per-token billing that makes it economically irrational to give the agent long context or let it retry, and those who have discovered that the moment a model sits behind a rate limit and a billing wall, their instinct is to use it less rather than more.

The per-token model is not a pricing quirk. It is an architectural choice that shapes what the agent can do. When every context token costs money, long sessions are expensive. When every retry costs money, the agent stops retrying. When every parallel draft costs money, you run one. When every background indexing call costs money, you don't run background indexing. The cloud model is optimized for the provider's margin. The ideal agent for a developer is optimized for the developer's throughput.

### What HIDE Is

**Hawking IDE (HIDE)** is a local-first agentic coding IDE. It runs entirely on your machine — no API calls, no telemetry, no model subscriptions. You download a Hawking model once and decode it indefinitely with your own Apple Silicon GPU, at zero marginal cost per token.

HIDE is the second product in the **Hawking family**. The first product, **Hawking Condense**, is the quantization and compression pipeline: it takes any HuggingFace model (7B, 32B, 72B, or beyond) and produces a `.tq` file — a trellis-quantized binary that fits a 32B model in roughly 18 GB of unified memory and decodes on the GPU at speeds that make multi-step agent loops practical. Condense is the manufacturing plant. HIDE is the end product that runs those models and uses them to write code.

The distribution model is:
1. You download HIDE (the app).
2. You either download a prebuilt Hawking model from the catalog (a `.tq` file hosted as a one-time download), or you paste any HuggingFace model URL and let Condense build one in the background.
3. You open your project and start using the agent. No account. No subscription. No data leaves the machine.

HIDE is not a thin wrapper around the OpenAI API with a local fallback. The local runtime *is* the product. The Hawking inference server (`hawking-serve`) ships inside the app, decodes `.tq` and GGUF weights, and exposes an OpenAI-compatible API surface to the agent layer above it. The agent layer (`hawking-agent`) is a Rust-native FSM loop — Planner → Executor → Verifier — that orchestrates tool calls, manages context, and checkpoints state to an append-only event log. The IDE surface (a Tauri 2 application) wraps both and provides the developer-facing UI: Chat, Diff Review, Context Stack, file tree, and integrated terminal.

### Who HIDE Is For

HIDE is for developers who:

- Work on codebases that are confidential, proprietary, or regulated — where sending source to a third-party API is a legal, contractual, or cultural non-starter.
- Want to run agent loops that are long, expensive, and parallel — the kind where a cloud subscription would cost hundreds of dollars a month.
- Want their model to improve over time at *their* codebase, not to be reset to a generic checkpoint every session.
- Work in environments without reliable internet (air-gapped facilities, airplanes, remote locations).
- Simply believe that a tool that processes their most sensitive intellectual property should not exfiltrate it by design.

### The "Exceed, Not Rival" Principle

HIDE does not aim to match Claude Code on Claude's home ground. Claude's home ground is access to Anthropic's most powerful frontier models, integration with the cloud, and breadth of supported workflows. On those axes, a local product cannot win and should not try.

HIDE exceeds cloud AI IDEs on the axes that are structurally inaccessible to cloud providers. It is not better in spite of being local — it is better *because* it is local. The capabilities enumerated in §0.2 are not compromises around a limitation; they are positive capabilities that emerge from the local runtime's architectural position. The product bet is that these capabilities are worth more to the target user than the delta in raw model capability.

### The Free-Forever Model

The Hawking runtime is open source. The Condense quantization pipeline is open source. The model weights are downloaded once from the Hawking catalog (or derived from public HuggingFace checkpoints) and stored locally. There is no subscription, no per-seat licensing, and no metered API. The business model, if any, sits at the layer of enterprise support, private model hosting, or specialized Condense runs — not at the token layer. A developer who downloads HIDE and a model will continue to have a working, capable coding agent indefinitely at zero recurring cost.

---

## §0.2 — The Local Superpowers

These are the eight capabilities that local-first buys that cloud providers fundamentally cannot replicate. Each is stated as: **capability → mechanism → concrete HIDE feature**.

### 1. Zero Per-Token Cost → Spend Lavishly

**Capability:** Every decode step is free at the margin. There is no economic pressure to use less context, run fewer retries, or constrain the number of parallel requests.

**Mechanism:** The model runs on your GPU. Compute is already paid for in the hardware purchase.

**HIDE features:** Context windows are set to the model's maximum by default, not to a cost-minimizing heuristic. The Planner is permitted to issue speculative tool calls in parallel without economic penalty. The Verifier can retry failed tasks with fresh context without triggering billing anxiety. Background codebase indexing, embedding generation, and re-ranking run continuously without throttling. Parallel agent drafts (ch.09) spin up multiple model instances simultaneously at zero marginal cost — the only constraint is RAM and thermal budget, both of which HIDE manages explicitly (see Superpower 6).

### 2. Direct Logit Access → Grammar-Constrained Decode

**Capability:** The host application has access to the raw probability distribution over the vocabulary at every decode step, before sampling. This is impossible in a cloud API where the model is a black box.

**Mechanism:** `hawking-serve` exposes logit masks through `json_constrain.rs::JsonConstraint::mask_logits`, which zeroes out any token that would produce invalid JSON (or any other grammar) at the current decode position.

**HIDE features:** Tool calls are schema-validated at the hardware decode level. The model *cannot* emit a malformed JSON tool call — invalid tokens are masked before sampling. This eliminates the entire category of post-hoc JSON repair, retry-on-parse-failure, and tool call hallucination that plagues cloud-API-backed agents. It also enables custom samplers: confidence heat-maps (surface tokens where the model is uncertain), structured diff generation (force the model to produce valid unified diffs), and grammar-constrained code generation (force syntactically valid output for a given language).

### 3. Full OS Ownership → Deterministic Sandbox

**Capability:** The agent process runs on the same machine as the developer's source tree, build tools, and runtime. The IDE controls the security boundary directly.

**Mechanism:** Apple's Seatbelt (`sandbox-exec`) provides a T0–T4 permission ladder (ch.10). The agent's tool execution context sits in a named sandbox profile that grants or denies file system access, network access, process spawning, and IPC on a per-task basis.

**HIDE features:** The agent can run builds, tests, linters, and shell commands with deterministic isolation. A T2-sandboxed build can write to the project directory and spawn child processes but cannot reach the network. A T1-sandboxed proof-of-concept can read files but cannot write them. The sandbox level is surfaced in the UI and adjustable per-workflow. Cloud agents run in the provider's container and cannot provide the same determinism guarantees about the local execution environment.

### 4. Durable Memory → Never-Reset Session History

**Capability:** The agent's state is not discarded between sessions. Every decision, every tool call result, every accepted and rejected diff is persisted.

**Mechanism:** An append-only event log (ch.01) records every `AgentEvent` with a wall-clock timestamp and a content-addressed payload. The log is stored on the local filesystem and is never truncated automatically.

**HIDE features:** The Context Compiler (ch.04) can reconstruct any prior session state by replaying the log. The agent remembers which files it has read, which edits were rejected, which tests failed and why, and what the user said three weeks ago. The Memory layer (ch.04) maintains a CoALA-style hierarchy: working memory (this task), episodic memory (this project's session history), semantic memory (embeddings of the codebase and past decisions), and procedural memory (the agent's fine-tuned priors). Cloud agents are reset to a blank context with every new session.

### 5. Personalization Flywheel → The Model Gets Better at Your Codebase

**Capability:** Because the model runs locally and Condense is available in-app, the local model can be continuously fine-tuned on the developer's interaction history.

**Mechanism:** Accepted diffs, approved tool call sequences, and explicit thumbs-up signals are written to a training record. Condense's LoRA-KD path (ch.11, `doctor_qat.py`) trains a LoRA adapter on this record and hot-swaps it into the serving model without restarting the server. The resulting model is re-baked into a new `.tq` checkpoint.

**HIDE features:** Over time, the local model learns the codebase's idioms, naming conventions, preferred libraries, test patterns, and the developer's review preferences. Each Condense re-bake produces a model that is measurably better at completing tasks in this specific codebase than a generic checkpoint. Cloud providers explicitly cannot do this: they cannot train on your private code, and even if they could, they cannot give you a model that has been tuned solely on your data.

### 6. Hardware Awareness → Thermal and RAM Governance

**Capability:** The host application knows exactly how much unified memory is installed, how much is currently in use, what the GPU and CPU temperatures are, and what the current thermal throttle state is.

**Mechanism:** macOS IOKit APIs expose thermal state, power metrics, and memory pressure in real time. `hawking-serve` queries these before accepting inference requests and adjusts decode parameters accordingly.

**HIDE features:** HIDE never OOMs the machine. The RAM governor caps total model memory footprint based on available headroom and adjusts which model roles are active (e.g., temporarily offloading the embedder when the hero model is running a long parallel draft). The thermal governor reduces batch size or decode throughput when the chip approaches its sustained power limit, preventing the machine from becoming unusable for other work. Cloud providers have no visibility into the user's hardware state and cannot provide these guarantees.

### 7. No Network Dependency → Air-Gap Mode

**Capability:** The entire inference stack, agent loop, tool set, and IDE surface operate without any internet connection.

**Mechanism:** All models are local `.tq` files. All tool execution is on the local filesystem or local processes. The Tauri app shell is a native binary with no cloud dependencies.

**HIDE features:** HIDE works on airplanes. It works in secure government facilities. It works during internet outages. It works in jurisdictions with restrictive data egress laws. The Research Lab (ch.08) can operate in a "local-only" mode that caches fetched sources and never makes outbound requests during a session. Optional network access for web research, documentation fetch, and model catalog updates can be individually toggled. The default is air-gap-safe.

### 8. Composable Model Roles → Multi-Model at Zero Marginal Cost

**Capability:** Multiple model instances can run simultaneously on the same machine, each specialized for a different role, with no per-request pricing for any of them.

**Mechanism:** `hawking-router` (ch.06) manages a pool of named model subprocess slots: `hero` (the primary coding model, typically 7B–32B), `draft` (a smaller fast model for speculative decode and quick answers), `embedder` (a small embedding model for semantic search), and `reranker` (a cross-encoder for retrieval ranking). Each slot is a separate `hawking-serve` process with its own model loaded.

**HIDE features:** Speculative decode pairs the hero model with the draft model to increase hero throughput. The embedder runs continuously in the background, indexing the codebase without waiting for the hero to be idle. The reranker improves retrieval quality for context selection without consuming hero compute. In a parallel agent workstation (ch.09), multiple hero instances run simultaneously. The total compute cost of all of these running simultaneously is the same: your hardware, already purchased.

---

## §0.3 — The Constitution

These are the twelve design principles that govern every architecture decision in the HIDE bible. They are ordered by how often they are likely to be invoked to resolve ambiguities. A contributor who reads these twelve principles and nothing else should be able to make the right decision in most novel situations.

**1. Control flow lives in Rust, not in the model.**

The agent loop is a deterministic finite state machine — `Idle → Planning → Executing → Verifying → Idle` — implemented in Rust (`hawking-agent/src/fsm.rs`). The model's role is to fill structured slots in that loop: produce a Plan, produce a ToolCall, produce a Verdict. It does not control the loop itself. There is no ReAct-style free-form reasoning chain where the model decides when to stop or what to do next; those decisions are in the FSM's transition logic. This is the single most important principle in the architecture. It is what makes the agent auditable, testable, and recoverable.

**2. Deterministic verifiers beat LLM critics.**

When a ground-truth verifier exists, use it. Build output (exit code 0/non-0), test results (pass/fail), linter output, and type-checker output are the primary Verifier signals. LLM-as-judge verdicts are used only when no deterministic verifier exists (e.g., "is this diff coherent?"). This ordering is not negotiable: the FSM's `Verifying` state always checks deterministic signals before consulting the model. The reason is that LLM critics have non-zero false-positive and false-negative rates at every capability level, while a compiler's exit code is exact.

**3. The event log is the system of record.**

All state is derived from a replay of the append-only event log. No in-memory data structure is authoritative; it is a cache derived from the log. This means: crash recovery is a log replay from the last checkpoint. Session restore is a log replay from session start. Debugging is a log replay. The log format is defined in ch.01 and must be stable — log entries are never rewritten or deleted (only compacted into checkpoint blobs). The consequence: any feature that "updates state" is actually "appends a state-change event."

**4. Deny beats allow, absolutely.**

In `PermissionPolicy`, a `deny` rule cannot be overridden by any `allow` rule at any scope. The evaluation order is: explicit deny → explicit allow → default deny. This applies at every layer: Seatbelt sandbox profiles, tool call authorization, MCP server capabilities, and file system access. The reason is that security properties must be composable — a plugin author must be able to reason that their `deny network` rule will hold regardless of what other plugins or the agent loop requests.

**5. Grammar-constrained decode is the default for structured output.**

Whenever the agent is asked to produce a structured output (tool calls, diff hunks, JSON payloads, commit messages), logit masking is the first choice, not post-hoc parsing or retry-on-error. The decode path in `hawking-serve` must expose a mask interface; callers must use it. Post-hoc JSON repair is permitted only as a fallback when the grammar is too complex to express as a finite-state mask (e.g., recursive structures). Any output that will be executed (shell commands, code edits) must go through the constrained decode path.

**6. The parity gate is sacred.**

GPU TQ decode output must be bit-identical to CPU oracle output on the parity test suite. Any divergence is a bug, not a warning, not a known issue, and not acceptable to ship. This principle exists because the entire quality story of HIDE depends on the Condense'd model being losslessly served. If GPU decode silently drifts from the expected distribution, all quality measurements are invalid. The parity test suite runs in CI; any commit that breaks it fails the build unconditionally.

**7. Shell-first, moonshots second.**

The core shell (agent kernel + router + Tauri app with Chat, Diff Review, Context Stack, terminal, file tree) must be complete and stable before any post-shell feature (Research Lab, Parallel Agent Workstation, RLEF, remote Mac Studio) is scheduled. Features in ch.08–11 exist in the bible because they are architecturally designed and have defined integration points; they are not scheduled. The consequence: if a shell feature and a moonshot feature compete for the same contributor bandwidth, the shell feature wins unconditionally.

**8. Open formats, open weights.**

HIDE serves `.tq` and GGUF model formats only. There is no integration with proprietary cloud model APIs, no dependency on a vendor-specific model format, and no feature that requires a non-open model to function. The `.tq` format is Hawking's open trellis-quantized format; GGUF is the community standard. The agent loop, tool system, and context compiler are model-agnostic: they interface with `hawking-serve`'s OpenAI-compatible API, which any compatible local server can satisfy. This is both a values commitment and an architecture requirement — no path in the codebase should have a hard dependency on a specific model or provider.

**9. The context budget is always explicit.**

Every call site that builds a context window must compute and log the token count before submitting the inference request. Context budget overflows are caught at the compiler level (the Context Compiler in ch.04 returns an error if the budget is exceeded) and handled by a defined truncation policy, not by silently cutting tokens at the API boundary. The truncation policy (recency + importance scoring) is deterministic and auditable. Silent truncation is a bug.

**10. No telemetry, air-gappable by default.**

No HIDE code path sends data outside the machine unless the user has explicitly enabled a feature that requires it (e.g., web research). There are no analytics, no crash reporters, no model improvement telemetry, and no usage logging to any remote endpoint. The default network posture is no outbound connections. Features that require the network must be individually enabled and must be clearly labeled in the UI. This is verifiable: the release build must pass a network-sandbox test that confirms no outbound connections are made during a typical agent session.

**11. Checkpoints are the unit of recoverability.**

The agent loop emits a `Checkpoint` event at every state transition boundary. A checkpoint contains enough information to resume the loop from that point without replaying the entire prior log. Checkpoints are written before any destructive action (file write, shell command, git commit). If the process crashes or is killed, the next startup replays from the last checkpoint forward. The consequence: any feature that introduces a new destructive action must define its checkpoint boundary. "Resume from before the action" must always be possible.

**12. Measure before you commit.**

No performance-sensitive architectural choice is finalized without a measured result on representative hardware (an M-series Mac with the target RAM tier). Claims of the form "this will be faster" or "this should reduce latency" are hypotheses, not decisions. The thesis gate (§0.4) is the most important instance of this principle, but it applies throughout: before choosing a context retrieval strategy, measure retrieval recall; before choosing a model role assignment, measure tok/s at the target sequence length; before choosing a sandbox profile, measure fork latency. The `hawking-eval` harness (ch.02) is the canonical measurement tool.

---

## §0.4 — Thesis Gate: The Load-Bearing Bet

### The Bet

HIDE's entire value proposition rests on a single empirical claim: a Condense'd 7B–32B `.tq` model, running on consumer Apple Silicon, can drive a multi-step agentic coding loop to successful completion at tok/s that make the experience usable.

"Usable" is defined concretely:
- The model must complete a representative coding task (a multi-file edit with tool calls, a failing test fixed, a refactor with verification) in under 10 minutes of wall-clock time.
- The decode throughput during the agent loop must be ≥ 15 tok/s on a 32B model on an M3 Pro (36 GB), or ≥ 25 tok/s on a 7B model on the same hardware.
- Task success rate on `hawking-eval` benchmark tasks must be ≥ 40% at 7B and ≥ 55% at 32B on first attempt (no human retry).

These numbers are not aspirational. They are the minimum bar below which HIDE does not deliver a meaningfully better experience than a developer doing the work manually.

### Gate Thresholds

| Verdict | Condition | Consequence |
|---|---|---|
| **GO** | All three criteria met at both 7B and 32B | Milestone 2 (Model Lab, Parallel Agents, Research) unlocked |
| **CONDITIONAL** | Throughput met; task success 30–40% at 7B or 45–55% at 32B | Milestone 2 deferred; Condense quality track escalated; re-gate in 4 weeks |
| **KILL** | Throughput < 15 tok/s at 32B OR task success < 30% at 7B | Stop HIDE shell development; reassess model strategy |

The KILL threshold is not a failure mode to plan around. It is a real possibility if the trellis quantization penalty at 3-bit or the GPU decode latency at 32B is worse than projected. The bible is written on the assumption that GO is achievable; the gate exists to surface CONDITIONAL or KILL before months of IDE shell work are sunk.

### The Measurement Harness

`hawking-eval` (not yet built at the time of writing; defined in ch.02) is the canonical harness. It runs a fixed set of coding tasks against a live `hawking-serve` instance, scores each task against a deterministic verifier (build output, test pass rate, diff correctness), and emits a structured JSON report. The gate measurement is run on a clean machine with no prior warm-up, using the standard Condense'd model for each size tier.

The gate measurement runs before any Milestone 2 feature is scheduled and before any Milestone 1 features are committed that depend on a GO verdict.

---

## §0.5 — How to Read This Bible

This document is build-ready engineering depth. It is not a product requirements document, a marketing brief, or a high-level architecture overview. Every chapter contains concrete schemas, state machine definitions, data structure layouts, and integration contracts. The level of detail is intentional: contributors should be able to implement a chapter without asking clarifying questions about the design.

**Reading order.** Start with ch.01 (System Architecture) and ch.02 (Agent Kernel). These two chapters define the foundational abstractions — the event log, the FSM, the tool interface, the permission model — that every other chapter builds on. After ch.01 and ch.02, chapters can be read in any order. Cross-references are explicit (e.g., "see ch.04 §4.3") and should be followed when the referenced section defines something the current section depends on.

**Deferred chapters.** Chapters marked `POST-SHELL` (ch.08, ch.09 Tier 2–4, ch.11) are fully designed but not scheduled. Their integration points are defined and their APIs are reserved; they are not implemented until the shell milestone is complete and the thesis gate passes. Reading them is useful for understanding the full design space; building them is gated.

**Versioning.** This bible is a living document. When a design decision changes, the relevant chapter is updated and the change is logged in the chapter's revision history footer. Do not treat any section as immutable; treat it as the current authoritative design until the footer says otherwise. The constitution (§0.3) is the most stable section; principles are changed only by explicit contributor consensus, not by individual PRs.

**What this bible is not.** It is not a user guide. It is not API documentation for external consumers. It is not a specification for the `.tq` format (that lives in the Condense codebase). It is the internal engineering reference for contributors building the HIDE shell and the agent stack above it.

---

## §0.6 — Scope: Initial Shell vs. Deferred

This section states clearly what is in scope for the initial HIDE shell build (Milestone 1) and what is deferred. The scope boundary is the input to project planning, not a wish list.

### In Scope — Initial Shell (Milestone 1)

**`hawking-agent` kernel.** The Planner → Executor → Verifier FSM. Plan-as-data (`AgentPlan` struct). Tool dispatch loop. Checkpoint and resume. The Oracle trait with build/test/lint verifiers. The skill library (read file, write file, run shell, search codebase, grep, git operations). The event log writer and reader.

**`hawking-router`.** Multi-model subprocess routing. Named model slots: `hero`, `draft`, `embedder`. Hot model swap (load/unload without restarting the router). OpenAI-compatible proxy interface.

**`.tq` GPU serving — completion.** The two remaining gaps in `hawking-serve`: LM-head TQ (first-class loader for the final linear layer in `.tq` format) and the first-class `.tq` model loader (currently partial). Both must be complete and parity-green before the thesis gate runs.

**Constrained-JSON tool calls.** The `JsonConstraint` mask interface wired end-to-end from `hawking-agent` through `hawking-serve`. Tool call schemas registered at server startup. All agent-issued tool calls go through the constrained decode path.

**Tauri 2 app shell.** Five UI regions: Chat (streaming message view, diff inline rendering), Diff Review (accept/reject hunks), Context Stack right-rail (visible token budget, active files, memory snippets), file tree, and integrated terminal (PTY, not a shell command passthrough). Basic settings panel (model selection, sandbox level, context budget).

**Codebase intelligence — baseline.** BLAKE3 file hashing, tree-sitter parse for Rust/TypeScript/Python, workspace symbol index, and the `SearchCodebase` tool backed by the index. Full stack-graphs and LSP integration are Milestone 2.

**`hawking-eval` harness — v1.** A minimal task runner sufficient to execute the thesis gate measurement. Fixed task set, deterministic verifier, JSON report output. Not the full eval suite; that grows in Milestone 2.

### Not In Scope — Deferred

**Hawking HF Org live catalog.** The public model catalog (download a prebuilt `.tq` from the Hawking HuggingFace org) is deferred until at least one 32B model has passed the thesis gate and is ready for public distribution. In Milestone 1, models are loaded from local paths.

**Research and Knowledge Lab (ch.08).** The multi-source research pipeline, knowledge graph, adversarial verifier, and Research Tab UI are POST-SHELL. The integration points are defined; the feature is not scheduled.

**Parallel Agent Workstation — Tier 2+ (ch.09).** Tier 1 (a single additional agent subprocess for background indexing) may ship in Milestone 1. Tiers 2–4 (full parallel workstation UI, merge funnel, remote Mac-Studio orchestration) are POST-SHELL.

**RLEF and on-device training (ch.11).** The personalization flywheel — training data collection, LoRA-KD re-bake, hot-swap — is a moonshot feature. The data collection hooks (logging accepted diffs) ship in Milestone 1 so the training corpus accumulates. The training pipeline and model re-bake are POST-SHELL.

**Model Lab and Store.** The in-app Condense pipeline (paste a HuggingFace URL, get a `.tq`) is Milestone 2, gated on the thesis GO verdict. In Milestone 1, Condense runs from the command line.

**Stack-graphs and full LSP integration (ch.05).** The baseline codebase intelligence (tree-sitter + symbol index) ships in Milestone 1. Full stack-graphs (cross-file name resolution, call graph) and the LSP-backed diagnostics feed are Milestone 2.

**Remote Mac-Studio (ch.09 Tier 4).** Distributed inference across multiple Apple Silicon machines over a local network. Fully designed; not scheduled until the single-machine stack is production-stable.

---

*End of chapter 00. Continue with ch.01: System Architecture.*
