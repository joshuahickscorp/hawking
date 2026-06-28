# 02 · The Agent Kernel

> **Purpose (one line).** The Agent Kernel is the deterministic brain *above* the model: a formally-specified, budget-governed, checkpoint-replayable state machine that turns a weak-but-free local model into a reliable software engineer by surrounding every generation with **plan-as-data**, **deterministic verification**, **bounded search**, and **durable self-correction** — so that when you build HIDE, the loop is *finished*, not started.

**Status:** DESIGN — flagship chapter. This is the definitive, final specification of the agent loop. It binds to the **Event envelope** (ch.01 §4.6), the **Extension manifest / capability negotiator** (ch.01 §7.2), the **ContextManifest / ContextSource / MemoryStore / KvStore** (ch.04 §4 + Appendix A), and the **runtime HTTP surface** (ch.01 §4.3). The model is a *stable localhost OpenAI-compatible service*; deeper model hooks (raw logits, constrained grammar, KV checkpoint) are designed here and tagged **[RUNTIME-SIDE — LATER]** with a **[SHELL-TODAY]** fallback that works against today's `/v1/chat/completions`, `/v1/hawking/generate`, `/v1/hawking/tokens`, `/v1/embeddings`, and `/metrics`.

---

## Table of contents

1. [Purpose & scope](#1-purpose--scope)
2. [Tenets](#2-tenets)
3. [State of the art + limits (cited)](#3-state-of-the-art--limits-cited)
4. [The Hawking design (concrete)](#4-the-hawking-design-concrete)
   - 4.1 [Module layout](#41-module-layout)
   - 4.2 [The loop as a formal state machine](#42-the-loop-as-a-formal-state-machine)
   - 4.3 [Budgets & the Governor](#43-budgets--the-governor)
   - 4.4 [Abort / interrupt / steer semantics](#44-abort--interrupt--steer-semantics)
   - 4.5 [Plan-as-data (HTN + dependency DAG)](#45-plan-as-data-htn--dependency-dag)
   - 4.6 [Verification: the reliability core](#46-verification-the-reliability-core)
   - 4.7 [Self-correction (Reflexion-style minimal repair)](#47-self-correction-reflexion-style-minimal-repair)
   - 4.8 [Search & sampling-scale strategies](#48-search--sampling-scale-strategies)
   - 4.9 [The tool-call protocol](#49-the-tool-call-protocol)
   - 4.10 [Subagents: spawn / delegate / return / isolate](#410-subagents-spawn--delegate--return--isolate)
   - 4.11 [The Skill Library (Voyager-style, persistent)](#411-the-skill-library-voyager-style-persistent)
   - 4.12 [Model-cooperation hooks](#412-model-cooperation-hooks)
   - 4.13 [Checkpoint / replay / resume](#413-checkpoint--replay--resume)
   - 4.14 [Multi-agent topologies & when single-agent wins](#414-multi-agent-topologies--when-single-agent-wins)
5. [How we EXCEED](#5-how-we-exceed-local-superpowers)
6. [Failure taxonomy → recovery matrix](#6-failure-taxonomy--recovery-matrix)
7. [Extensibility / plugin points](#7-extensibility--plugin-points)
8. [Bleeding-edge / moonshots (ranked)](#8-bleeding-edge--moonshots-ranked)
9. [Open questions / dials](#9-open-questions--dials)
10. [Cross-references](#10-cross-references)
- [Appendix A — Binding contracts (schemas other chapters import)](#appendix-a--binding-contracts)
- [Appendix B — Source register](#appendix-b--source-register)

---

## 1. Purpose & scope

This chapter specifies the **control system** that drives the model. It is the answer to a single design question:

> *Given a model that is smaller and weaker than Claude, with no per-token cost and full local access, how do we make it reliable enough to ship code?*

The thesis, stated once and defended throughout: **reliability comes from deterministic verification plus bounded search, not from model cleverness.** A 7B model that proposes a wrong patch is cheap to catch (the build fails, the test fails, the patch won't apply) and cheap to retry (no bill, no rate limit). The kernel's job is to convert the model's *fallible single attempt* into a *verified outcome* by wrapping it in oracles and search. Where a cloud agent rations attempts because each costs money, HIDE spends lavishly: many parallel drafts, exhaustive verification, tree search over tool trajectories, overnight runs that resume across days.

### In scope

- The **formal loop**: an explicit state machine `INTAKE → PLAN → SELECT_STEP → ACT → OBSERVE → VERIFY → {DONE | REPAIR | REPLAN} → FINALIZE`, with complete transition tables, guards, and effects.
- **Budgets & the Governor**: max_steps / retries / replans / tokens / wallclock / cost-equivalent, enforced centrally, with abort/interrupt/steer.
- **Plan-as-data**: a hierarchical task network with a typed JSON plan schema, a dependency DAG of steps, and replanning semantics.
- **Verification**: deterministic oracles (patch-apply, build, typecheck, lint, test, grep/AST) + LLM self-check fallback + process-reward/self-consistency.
- **Self-correction**: Reflexion-style minimal, failure-only repair prompts and a durable lessons store.
- **Search & sampling-scale**: ReAct baseline, best-of-N, Tree-of-Thoughts, MCTS/LATS over tool trajectories, debate/critic panels — and the *escalation policy* deciding which to use.
- **Tool-call protocol**: the request/response envelope, capability binding, constrained-decode emission, idempotency, and event mapping.
- **Subagents**: spawn, delegation contract, isolation, return-summary protocol.
- **Skill Library**: a persistent, versioned, retrievable store of learned skills/recipes that survives sessions.
- **Model-cooperation hooks**: constrained-decode for plans/tool-calls, logit-confidence gating, entropy-triggered escalation, speculative self-drafting.
- **Checkpoint / replay / resume**: durable, deterministic, resumable-across-days agent state.
- **Failure taxonomy → recovery**: the catalogue of how agents fail and the matrix that handles each.
- **Multi-agent topologies** and the rule for when a single agent is correct.

### Out of scope / deferred (with gates)

| Item | Where it lives | Status |
|---|---|---|
| Diff/merge UX, the plan-tree UI surface | ch.03 editor | **Consumes** our `plan.*`/`diff.*` events; we define the data, ch.03 renders it. |
| Context packing, retrieval ranking, memory schemas | ch.04 | **Binding contract.** We *request* context via the Context Compiler and *read/write* `MemoryStore`; we never re-implement them. |
| Sampler/grammar/KV kernels, `.tq`/32B serving | ch.06 + *Hawking Condense* | **[RUNTIME-SIDE — LATER].** Designed here at each hook; the loop runs against today's HTTP surface without them. |
| Tool *implementations* (the actual `fs.write`, `shell.run` bodies) | ch.01 §7 tool registry + ch.04 | We define the **protocol & dispatcher**; tools are extensions. |

> **Scoping invariant.** Every subsection that reaches *inside the model* carries a **[RUNTIME-SIDE — LATER]** tag and a **[SHELL-TODAY]** fallback. The shell — including the *entire* agent loop — is never blocked on a kernel landing. The loop is correct on plain HTTP and merely *better* with the deeper hooks.

### Ground truth this chapter binds to (verified in-tree)

The runtime already exposes the seams this chapter drives. Bound to **real types**, not aspirations:

- **`hawking-core/src/engine.rs`** — `trait Engine { fn generate(&mut self, req: GenerateRequest, sink: &mut dyn FnMut(StreamEvent)) -> Result<GenStats>; … }`. `StreamEvent::{Token{id,text}, Done{reason,stats}}`. `GenStats::dec_tps()`. `GenerateRequest { prompt, max_new_tokens, sampling, stop, abort: Option<Arc<AtomicBool>>, max_stall_ms, json_mode }`. `SamplingParams { temperature, top_k, top_p, repetition_penalty, seed: Option<u64> }` — **`seed` is the determinism handle**. `StopReason::{MaxTokens, StopString, Eos, Aborted}` — **`Aborted` is the cooperative-cancel signal** (Ctrl-C / HTTP cancel / per-token `max_stall_ms` watchdog). `SpeculateMode::{Off, ExactShared, Eagle5}` with **greedy bit-identity at temp=0** — the self-drafting hook (§4.12). `model_arch()` — routes transformer vs SSM.
- **`hawking-core/src/json_constrain.rs`** — `JsonConstraint::mask_logits(&self, vocab, logits)` masks the next-token distribution; `json_mode` on the request triggers it. **This is the constrained-decode substrate for plans and tool-calls** (§4.9, §4.12), extensible from JSON to JSON-Schema/grammar.
- **`hawking-serve/src/http.rs`** — routes `/v1/chat/completions` (SSE), `/v1/completions`, `/v1/models`, `/v1/embeddings`, native `/v1/hawking/generate` (lean SSE), `/v1/hawking/tokens` (raw token-id SSE — minimum overhead, the loop's hottest path), `/healthz`, `/metrics`. The native `/v1/hawking/generate` final event carries `{stats:{dec_tps}}`.
- **`hawking-serve/src/spec_gov.rs`** — a rolling acceptance tracker that auto-enables/disables speculation by measured accept rate. The kernel reads its `/metrics` counters into the Governor's telemetry (§4.3).
- **`hawking-core/src/stateful/`** — `KvEvictionPolicy`, `prefix_cache.rs`, `attn_capture.rs`; **`hawking-serve/src/system_kv_bank.rs`** prefix-KV reuse. These are ch.04's; the kernel *consumes* them via the Context Compiler and the KV checkpoint hook (§4.13).

Everything tagged **[SHELL-TODAY]** has been checked to run against the routes above. Everything tagged **[RUNTIME-SIDE — LATER]** names the exact seam (e.g. a `logprobs` field on the SSE token event, a `POST /v1/hawking/kv/checkpoint` route) and ships a fallback.

---

## 2. Tenets

Twelve tenets. Every later decision cites one. These extend ch.01's T1–T10 and ch.04's tenets into the loop's domain.

| # | Tenet | Consequence |
|---|-------|-------------|
| **K1** | **Verification is the product; generation is a proposal.** A model output is never trusted until an oracle confirms it. | The loop's center of gravity is `VERIFY`, not `ACT`. No state advances on faith (§4.6). |
| **K2** | **Determinism over cleverness.** Reliability is bought with deterministic oracles + bounded search, not by hoping the model is smart. | Small models win when the *harness* is rigorous. Same log + same seeds ⇒ same trajectory (§4.13). |
| **K3** | **The plan is data, not prose.** The agent's intent lives in a typed, inspectable, editable plan object — never buried in a chat transcript. | Plans are events, diffable, user-editable, replan-able, and a DAG (§4.5). |
| **K4** | **Spend lavishly, locally.** No per-token cost, no rate limit → best-of-N, tree search, self-consistency, overnight runs are *defaults under a budget*, not luxuries. | The Governor caps wallclock/steps, *not* token spend, by default (§4.3). The escalation ladder (§4.8) is the spend policy. |
| **K5** | **Effects are recorded outcomes; replay never re-fires them.** Every tool call, file write, and generation is recorded as its observed result and replayed *as data* (ch.01 T3). | The loop is replay-safe by construction. Forward-resume is the only path to new effects (§4.13). |
| **K6** | **Minimal-context repair.** On failure, the agent retries with the *smallest* high-signal failure context (the diff, the error, the lesson) — not the whole bloated history. | Reflexion, made mechanical and minimal (§4.7). Fights context rot (ch.04 §6). |
| **K7** | **The model is a cooperating party.** We read its logits (confidence/entropy), constrain its decode (valid plans/tool-calls by construction), and draft against it (spec-decode). | Cloud sees a sealed box; we co-design (§4.12). |
| **K8** | **Bounded everything.** Every loop, search, retry, subagent fan-out, and edit count has a hard ceiling enforced centrally. | No runaway agents. The Governor is the single chokepoint (§4.3). Infinite-edit and loop failures are *structurally impossible*, not merely discouraged. |
| **K9** | **Isolation by default for parallel work.** Parallel attempts and subagents run in isolated workspaces/contexts; results merge through verification, never by trampling shared state. | Worktree-per-attempt; clean-window subagents return summaries (§4.10). |
| **K10** | **Skills compound.** A verified solution is captured as a reusable, retrievable skill that survives sessions, so the agent gets monotonically better at *this* repo. | The Skill Library is a first-class store, not a cache (§4.11). |
| **K11** | **Every decision is an event.** Plan steps, tool calls, verifications, repairs, escalations, and budget transitions all emit ch.01 envelope events. | The whole trajectory is auditable, replayable, and resumable (Appendix A; ch.01 §4.6). |
| **K12** | **Single-agent is the default; multi-agent earns its place.** Coordination adds failure modes (§3); we reach for it only when the task is genuinely separable and verifiable per-part. | §4.14's rule gates every topology. |

---

## 3. State of the art + limits (cited)

Tagged **[PROVEN]** (deployed/replicated at scale) / **[RESEARCH-PROVEN]** (strong published results, less battle-tested) / **[SPECULATIVE]**, with *difficulty* (build cost for us) and *impact*. Full register in [Appendix B](#appendix-b--source-register).

### 3.1 Core reasoning/acting loops

| Pattern | Mechanism (compressed) | Headline | Maturity | Diff / Impact |
|---|---|---|---|---|
| **ReAct** | Interleave `Thought → Action → Observation`; reasoning conditions tool use and tool results condition reasoning. | The baseline agent loop; reduces hallucination vs reason-only by grounding in observations. | [PROVEN] (every coding agent uses it) | low / foundational. ([Yao et al. 2023]) |
| **Reflexion** | After a failed attempt, the model writes a *verbal self-reflection* stored as episodic memory and prepended to the next attempt. | **HumanEval 91%** (> GPT-4's 80% at the time); +22% AlfWorld, +20% HotPotQA. | [RESEARCH-PROVEN] | low / **high** — our self-correction core (§4.7). ([Shinn et al. 2023]) |
| **Plan-and-Execute / Plan-and-Act** | A planner LLM emits a high-level plan; an executor LLM runs steps with *localized* replanning. | Plan-and-Act: **57.6%** WebArena-Lite (SOTA at pub); dynamic replanning **+34 pp** over ReAct. | [RESEARCH-PROVEN] | med / high — our plan layer (§4.5). ([Erdogan et al. 2025]) |
| **Tree-of-Thoughts (ToT)** | Explore a tree of intermediate thoughts; evaluate states; BFS/DFS to the best leaf. | Game-of-24 4% (CoT) → **74%** (ToT). | [RESEARCH-PROVEN] | med / high for *branchy* decisions (§4.8). ([Yao et al. 2023b]) |
| **Self-Consistency** | Sample N reasoning paths, take the majority answer. | +17.9 pp GSM8K over greedy CoT. The cheapest test-time-scaling win. | [PROVEN] | low / high — N parallel attempts → vote (§4.8). ([Wang et al. 2023b]) |
| **LATS** | MCTS over `(reason, act, observe)` nodes; value = LM self-eval + environment reward; reflection on failed rollouts. | Beats ReAct/ToT/Reflexion/RAP on programming + web; **HumanEval 92.7%** with GPT-4. | [RESEARCH-PROVEN] | high / high — our deepest search tier (§4.8). ([Zhou et al. 2024]) |
| **RAP / MCTS-for-reasoning** | World-model-guided MCTS; the LLM is both policy and value. | Strong on planning/math vs CoT. | [RESEARCH-PROVEN] | high / med. ([Hao et al. 2023]) |

### 3.2 Verification, reward, and selection

| Pattern | Mechanism | Headline | Maturity | Diff / Impact |
|---|---|---|---|---|
| **Best-of-N + verifier** | Sample N candidates; a *verifier* (oracle or reward model) re-ranks; take the top. | The dominant test-time-scaling shape; verifier quality is the ceiling. | [PROVEN] | low / high (§4.8). ([Cobbe et al. 2021]) |
| **Process Reward Models (PRM)** | Score *each reasoning step*, not just the final answer; select trajectories with high stepwise reward. | Qwen2.5-Math-PRM, ReasonFlux-PRM improve Best-of-N selection; step-level >> outcome-only on hard reasoning. | [RESEARCH-PROVEN] | med–high (needs a PRM) / high. ([Zhang et al. 2025 PRM survey]) |
| **Self-PRM / Generative verifier** | The LLM verifies its own steps ("Process Reward Models That Think") — a verifier *is a generator* prompted to critique. | Approaches trained-PRM quality with no separate model. | [RESEARCH-PROVEN] | low (prompt-only) / med — our LLM-judge fallback (§4.6). ([Zhao et al. 2025]) |
| **LLM-as-Judge** | An LLM scores/critiques an output against a rubric. | Ubiquitous for eval; *biased & gameable* — must be a fallback to deterministic oracles, never primary for code. | [PROVEN, with caveats] | low / med (§4.6 — strictly *after* deterministic oracles). ([Zheng et al. 2023b]) |
| **Tool-integrated self-verification (T1)** | Small models verify with *tools* (run the code, check the type) rather than introspection. | Lifts *small*-model test-time scaling specifically — directly our regime. | [RESEARCH-PROVEN] | low / **high** — validates our oracle-first thesis for 7B. ([Kang et al. 2025]) |
| **Agentic rubrics as verifiers** | Generate task-specific rubrics, then check the candidate against them. | Improves SWE-agent acceptance via contextual verification. | [RESEARCH-PROVEN] | med / med (§4.6 rubric oracle). ([2026 rubric-verifier]) |

> **Limit to internalize (the verifier-quality ceiling).** Best-of-N and PRM selection are *only as good as the verifier*. A reward model can be gamed; an LLM judge is biased and inconsistent ([bias-amplification in LLM-judge, 2025]). **Therefore HIDE ranks oracles strictly: a deterministic oracle (the compiler/test/patch-apply *ran*) always outranks any model-based score.** Model scores break ties *among oracle-passing candidates*, never override an oracle.

### 3.3 Coding-agent harnesses (the direct competitors)

- **SWE-agent** — introduced the **Agent-Computer Interface (ACI)**: a small set of LM-friendly commands (`open`, `edit`, `search`, `submit`) with guardrails (e.g. an edit-linter that rejects syntactically broken edits before they apply). Lesson: *constrain the action space and lint every edit at the boundary.* ([Yang et al. 2024]) → our tool-call linting (§4.9) and edit guardrails.
- **OpenHands** — the reference *event-sourced* agent: `Event/Action/Observation`, an append-only `EventStream`, replay that re-runs Actions and regenerates Observations, an anti-replay-loop guard (reject events with an existing id). **72% on SWE-Bench Verified (Sonnet 4.5).** ([Wang et al. 2025 OpenHands SDK]) → ch.01 adopted its event model; our loop *emits* into it.
- **Aider** — repo-map + direct file edits + **git as the event log** (auto-commit each AI edit, `/undo` reverts the last). Lesson: *cheap, legible checkpoints*; limit: conflates user VCS with agent history. ([aider docs]) → our checkpoints are richer (KV + manifest + plan), git-separate (ch.01 §4.5/§4.13).
- **Cline / Roo** — VS Code agents with explicit *plan-mode vs act-mode*, human-in-the-loop approval per tool call, and a checkpoint timeline. Lesson: *separate planning from acting and gate effects on approval.* → our autonomy levels (§4.3) + plan/act split (§4.2).
- **Context-as-a-tool for long-horizon SWE** (2025) — long-horizon agents need *active context management* (compaction, sub-task memory) to not collapse. ([Context-as-a-Tool 2025]) → binds ch.04; our subagent isolation + per-step context requests (§4.10).
- **Subtask-level memory for SWE agents** (2026) — *structurally aligned* memory keyed to subtasks improves SWE agents. → our episodic memory keyed by plan-step (§4.7, ch.04).

### 3.4 Skill libraries & procedural memory

- **Voyager** — generates **code skills**, validates them by *execution*, and stores verified skills in a library indexed by natural-language description; retrieves by embedding similarity; lifelong improvement with no gradient updates. ([Wang et al. 2023c]) → our Skill Library (§4.11) is Voyager-for-code-tools.
- **Memp / hierarchical procedural memory** (2025) — distills trajectories into reusable procedures; Bayesian selection + contrastive refinement of which procedures to keep. ([Memp 2025]) → our skill-promotion & decay policy (§4.11).
- **Generative Agents** — the retrieval score `recency + importance + relevance` (each ∈ [0,1]) that ch.04 adopts; we reuse the same shape to rank *skills* and *lessons*.

### 3.5 Multi-agent — and its sobering limits

- **AutoGen / society-of-mind, debate** — multiple agents converse/critique. ([Wu et al. 2023], [Du et al. 2023])
- **The 2025 reckoning.** Multi-agent debate **fails to consistently beat single-agent test-time compute at equal token budget**; *majority pressure suppresses independent correction* (agents conform rather than deliberate). ([Multi-LLM-Agents-Debate ICLR-blog 2025], [Tran & Kiela 2025]) Single-agent often wins multi-hop reasoning under an equal thinking budget.
- **MAST failure taxonomy** (NeurIPS 2025, 1,600+ traces) — 14 multi-agent failure modes in 3 buckets: **specification/design 41.8%**, **inter-agent misalignment 36.9%**, **verification gaps 21.3%**. ([Cemri et al. 2025]) → §4.14 reads this as: *most multi-agent failure is bad task specification and missing verification — both of which our plan-as-data + oracle-first design already attack in the single-agent case.*

> **Synthesis that sets HIDE's policy.** The literature is loud on two points: (1) **test-time compute scales reliability** (self-consistency, best-of-N, ToT/LATS) — and we have *free* test-time compute; (2) **the verifier is the ceiling** and **multi-agent coordination is often a net negative**. So HIDE's strategy is: *single agent + deterministic oracle-first verification + aggressive single-agent test-time search, escalating multi-agent only for genuinely separable, independently-verifiable work.* This is the spine of §4.

### 3.6 Constrained decoding for protocol reliability

- **Grammar/JSON-Schema constrained decoding** masks invalid next-tokens to `-∞`, making malformed tool-calls **architecturally impossible** (vs 1–5% parse-error rates from prompt-only structured output). XGrammar/llguidance achieve near-zero overhead; tag-triggered structure switching constrains tool-name→args. ([XGrammar-2 2026], [constrained-decoding refs]) → §4.9/§4.12: HIDE emits *plans and tool-calls under constraint* via the in-tree `JsonConstraint::mask_logits`, extended to schema/grammar. **This is a local superpower cloud APIs only partially expose.**

---

## 4. The Hawking design (concrete)

### 4.1 Module layout

The kernel is a headless Rust crate (`hide-kernel`, hosted in-process by the Tauri host per ch.01 §4.1). It depends on the *HTTP surface* of `hawking-serve` (via a runtime client), the **ch.01 event log/bus**, and the **ch.04 `hawking-context`** crate (Context Compiler + MemoryStore). It links **no** GPU code.

```
hide-kernel/                        # the brain above the model (headless, unit-testable)
  src/
    lib.rs                          # AgentKernel: owns sessions, runs, the governor
    machine/                        # THE LOOP
      state.rs                      # AgentState enum + Phase + transition table (§4.2)
      driver.rs                     # the step() executor: one transition per call, replay-safe
      guards.rs                     # transition guards (budget ok? verified? deps met?)
      effects.rs                    # effect emission → events (never run during replay) (K5)
    plan/                           # PLAN-AS-DATA (§4.5)
      schema.rs                     # Plan, Step, StepKind, DependencyDag (Appendix A.1)
      planner.rs                    # constrained-decode plan synthesis (§4.12)
      replan.rs                     # localized + full replanning policy
      dag.rs                        # topological order, ready-set, cycle detection
    verify/                         # VERIFICATION (§4.6)
      oracle.rs                     # Oracle trait + Verdict (Appendix A.2)
      deterministic/                # patch_apply.rs build.rs typecheck.rs lint.rs test.rs grep_ast.rs
      llm_judge.rs                  # LLM self-check FALLBACK (gated below deterministic)
      consistency.rs                # self-consistency vote + PRM-style scoring
      gate.rs                       # the VerificationGate: ranks oracles, decides DONE/REPAIR
    search/                         # SEARCH & SAMPLING-SCALE (§4.8)
      strategy.rs                   # Strategy trait + EscalationLadder
      react.rs  best_of_n.rs  tot.rs  lats.rs  debate.rs
      score.rs                      # candidate scoring (oracle ⊕ model ⊕ confidence)
    tools/                          # TOOL-CALL PROTOCOL (§4.9)
      protocol.rs                   # ToolCall / ToolResult envelopes (Appendix A.3)
      dispatcher.rs                 # capability-checked dispatch → ch.01 tool registry
      idempotency.rs                # dedup, idempotency keys, replay short-circuit
      lint.rs                       # pre-flight tool-call validation (SWE-agent ACI lesson)
    subagent/                       # SUBAGENTS (§4.10)
      spawn.rs                      # SubagentSpec, isolation (worktree/context)
      protocol.rs                   # delegation request + return-summary contract (Appendix A.4)
      registry.rs                   # live subagent tracking, budgets, cancellation
    skills/                         # SKILL LIBRARY (§4.11)
      library.rs                    # Skill schema, store (SQLite+vec via ch.04 MemoryStore)
      retrieve.rs                   # embed-indexed retrieval; recency/importance/relevance
      curate.rs                     # capture-on-success, promote/decay, contrastive refine
    govern/                         # BUDGETS & CONTROL (§4.3, §4.4)
      governor.rs                   # Budget, Ledger, enforcement, telemetry from /metrics
      interrupt.rs                  # abort/pause/steer wiring (→ GenerateRequest.abort)
      autonomy.rs                   # suggest-only ↔ auto-apply policy + approval gates
    cooperate/                      # MODEL-COOPERATION (§4.12)
      confidence.rs                 # logprob/entropy gating  [RUNTIME-SIDE — LATER hooks]
      constrain.rs                  # schema/grammar emission via JsonConstraint
      draft.rs                      # speculative self-drafting control (SpeculateMode)
    runtime_client.rs               # HTTP client → /v1/hawking/generate|tokens, /chat, /embeddings
    checkpoint.rs                   # agent-state snapshot/restore (§4.13) — folds on the event log
```

> **Why headless + crate-bounded.** The loop must be testable with **no GPU and no model** — we mock the runtime client with a scripted token stream and assert the exact event sequence. This is how "the loop is finished" becomes a *checkable* claim: a property-test suite over the state machine, the budget governor, the verifier gate, and replay determinism (§9 lists the suite). It is also reusable by a future CLI (`hide run --headless`).

---

### 4.2 The loop as a formal state machine

The agent is a **finite-state machine with an explicit stack** (for subagents and nested search) and a **budget ledger** (the Governor). One `driver.step()` call performs **exactly one transition**, emits its events, and returns — so the loop is *interruptible at every boundary* (§4.4) and *replayable transition-by-transition* (§4.13, K5).

#### 4.2.1 States

```rust
enum Phase {
    Intake,        // parse the user turn; load context; decide trivial-vs-plan
    Plan,          // synthesize/repair the plan-as-data (HTN + DAG)
    SelectStep,    // pick the next ready step from the DAG (or conclude all done)
    Act,           // execute the step: generate (maybe under search) → tool-call(s)
    Observe,       // ingest tool results / generation into working state
    Verify,        // run oracles on the step's outcome → Verdict
    Repair,        // minimal-context self-correction for a failed step
    Replan,        // revise the plan (localized or full) when steps are wrong
    Finalize,      // assemble the answer, write memory/skills, summarize
    Done,          // terminal success
    Aborted,       // terminal: user-cancel / budget-exhausted / fatal
    Paused,        // suspended awaiting human approval or input (resumable)
}

struct AgentState {
    phase: Phase,
    run_id: RunId,
    plan: Plan,                       // §4.5 — the live DAG (empty until Plan)
    cursor: Option<StepId>,           // current step under Act/Verify/Repair
    ledger: BudgetLedger,             // §4.3 — consumed vs caps
    stack: Vec<Frame>,                // search nodes & subagent frames (bounded depth)
    last_verdict: Option<Verdict>,    // §4.6
    repair_count: BTreeMap<StepId, u8>,
    replan_count: u8,
    pending_approval: Option<ApprovalRequest>,  // set when Paused
    context_manifest: Option<ManifestRef>,      // ch.04 — what the model saw this step
}
```

#### 4.2.2 Transition table (normative)

`g:` = guard (must hold to take the edge). `fx:` = effects (events emitted; ch.01 envelope kinds in `code`).

| From | Event/Trigger | Guard | To | Effects |
|---|---|---|---|---|
| `Intake` | turn admitted | always | `Plan` | `turn.assistant_started`; request context (ch.04) → `context.update`; classify complexity |
| `Intake` | trivial turn (one-shot, no tools) | `complexity == trivial && autonomy allows` | `Act` | skip planning; mark single implicit step |
| `Plan` | plan synthesized | `plan.valid() && dag.acyclic()` | `SelectStep` | `plan.step`×N (one per step), `plan.created` |
| `Plan` | plan synthesis failed/invalid | `replan_count < max_replans` | `Plan` | `error{plan.invalid}`; re-prompt with schema constraint (§4.12) |
| `Plan` | plan synthesis failed | `replan_count >= max_replans` | `Aborted` | `error{plan.unrecoverable, fatal}`, `turn.assistant_ended{stop=failed}` |
| `SelectStep` | a ready step exists | `∃ step: deps satisfied ∧ status=pending` | `Act` | `plan.step_updated{active}`; set `cursor`; request step context (ch.04) |
| `SelectStep` | no ready steps, all done | `dag.all(done\|skipped)` | `Finalize` | — |
| `SelectStep` | no ready steps, some blocked | `∃ failed step with no repair left` | `Replan` | `error{plan.blocked}` |
| `SelectStep` | step needs human approval | `autonomy == suggest-only ∧ step.effectful` | `Paused` | `approval.requested`; suspend (resumable) |
| `Act` | generation+tool-calls done | `Governor.ok()` | `Observe` | `token`/`token_batch`×, `tool.call`×, `tool.result`× (Observation-class) |
| `Act` | budget exhausted mid-step | `!Governor.ok()` | `Aborted` | `error{budget.exhausted, fatal}` |
| `Act` | invalid tool-call (lint reject) | `lint fails ∧ retries left` | `Act` | `error{tool.invalid}`; re-prompt under constraint (§4.9) |
| `Observe` | results ingested | always | `Verify` | `context.update` (tool outputs folded); update working state |
| `Verify` | oracles pass | `gate.verdict == Pass` | `SelectStep` | `verify.result{pass}`, `plan.step_updated{done}`; maybe `diff.applied` |
| `Verify` | oracles fail, repair budget left | `verdict == Fail ∧ repair_count[step] < max_repairs` | `Repair` | `verify.result{fail, failures[]}` |
| `Verify` | oracles fail, no repair left | `verdict == Fail ∧ repair_count[step] >= max_repairs` | `Replan` | `verify.result{fail}`, `plan.step_updated{failed}` |
| `Verify` | inconclusive (oracle absent) | `verdict == Inconclusive` | `Repair`/`SelectStep` | `verify.result{inconclusive}`; LLM-judge fallback decides (§4.6) |
| `Repair` | repair plan formed | always | `Act` | `repair.attempt{step, lesson_ref}`; `repair_count[step] += 1`; minimal-context prompt (§4.7) |
| `Replan` | localized replan succeeds | `localized ∧ replan_count < max` | `SelectStep` | `plan.step`×(new/changed), `plan.replanned{scope=local}`, `replan_count += 1` |
| `Replan` | full replan | `replan_count < max` | `Plan` | `plan.replanned{scope=full}` (carries lessons forward) |
| `Replan` | replan budget exhausted | `replan_count >= max` | `Finalize` | `error{plan.exhausted}`; finalize with partial result + honest report |
| `Finalize` | success path | `goal satisfied` | `Done` | write episodic memory + skills (§4.11), `turn.assistant_ended{stop=done}`, summary |
| `Finalize` | partial/failed path | `!goal satisfied` | `Done` | honest partial report + `repair.lesson` written; `turn.assistant_ended{stop=partial}` |
| `Paused` | approval granted | user intent | `Act`/`SelectStep` | `approval.granted`; resume from saved frame |
| `Paused` | approval denied / steer | user intent | `Replan`/`Plan` | `approval.denied`; fold user steer into plan |
| *any* | abort intent / fatal | always | `Aborted` | flip `GenerateRequest.abort`; `run.aborted`; checkpoint state (§4.13) |

**ASCII view of the happy path + the three loops:**

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │                         (abort / fatal → ABORTED, checkpointed)         │
   ▼                                                                        │
INTAKE ─▶ PLAN ─▶ SELECT_STEP ─▶ ACT ─▶ OBSERVE ─▶ VERIFY ──pass──▶ (next) ─┘
            ▲          │  ▲                            │
            │          │  │                            ├── fail, repair-left ──▶ REPAIR ─▶ (re-ACT)
   full     │       all │  └────────────────────────  │                                    │
   replan   │       done│                              ├── fail, no-repair ────▶ REPLAN ───┘
            │           ▼                              │                          │ (local→SELECT,
            └────────REPLAN◀───────────────────────────┘                          │  full→PLAN)
                                                                                  ▼
                                          SELECT_STEP(all done) ─▶ FINALIZE ─▶ DONE
                  (suggest-only + effectful step) ─▶ PAUSED ⇄ resume
```

Three nested loops, each independently budgeted (§4.3): **(a) the step loop** `SELECT_STEP→ACT→OBSERVE→VERIFY` (one per plan step); **(b) the repair loop** `VERIFY→REPAIR→ACT` (bounded by `max_repairs` *per step*); **(c) the replan loop** `…→REPLAN→{SELECT_STEP|PLAN}` (bounded by `max_replans` *per run*). Search (§4.8) nests *inside* `ACT` as a sub-machine on the `stack`.

#### 4.2.3 Driver contract (pseudocode)

```rust
/// One transition. Pure-ish: reads state, may call the runtime/tools/oracles,
/// emits events, returns the next state. NEVER advances on un-verified output (K1).
/// During REPLAY, `mode == Replay` and `effects::run_*` are short-circuited to
/// fold the RECORDED outcome instead of re-firing (K5).
fn step(state: &mut AgentState, env: &Env, mode: Mode) -> StepOutcome {
    govern::check(state, env)?;                       // K8: budget gate before any work
    match state.phase {
        Intake     => intake(state, env),
        Plan       => plan::synthesize_or_repair(state, env, mode),
        SelectStep => dag::pick_ready(state),         // or → Finalize / Replan
        Act        => search::run_step(state, env, mode),   // §4.8 may branch best-of-N/ToT/LATS
        Observe    => observe::ingest(state, env, mode),
        Verify     => gate::verify(state, env, mode), // §4.6 — the reliability core
        Repair     => repair::form_minimal(state, env),    // §4.7
        Replan     => replan::revise(state, env, mode),
        Finalize   => finalize::assemble_and_learn(state, env),  // §4.11 capture skill/lesson
        Paused | Done | Aborted => StepOutcome::Halt, // driven by external intents
    }
}

/// The run loop (driven by a session worker, ch.01 §4.9). Cooperative & resumable.
fn run(state: &mut AgentState, env: &Env) {
    while !state.phase.is_terminal() {
        if env.interrupt.poll() { handle_interrupt(state, env); }   // §4.4 — every boundary
        match step(state, env, Mode::Live) {
            StepOutcome::Continue => checkpoint::maybe(state),      // §4.13 — durable progress
            StepOutcome::Halt     => break,
            StepOutcome::Error(e) => govern::on_error(state, e),    // → Repair/Replan/Aborted
        }
    }
}
```

**Why one-transition-per-call.** It makes the loop (1) *interruptible* — abort/pause/steer are checked between transitions, so cancellation is bounded by one transition's latency, not a whole turn; (2) *replayable* — replay folds the recorded events transition-by-transition with effects disabled (K5); (3) *checkpointable* — `AgentState` is a small serializable struct snapshot-able after any transition (§4.13); (4) *testable* — a property test drives `step()` over a scripted runtime and asserts the exact `(phase, events)` sequence. This is the structural reason the loop can be declared *final*: its behavior is a finite, enumerable, test-covered transition relation.

---

### 4.3 Budgets & the Governor

The Governor is the **single chokepoint** (K8) that makes runaway agents *structurally impossible*. Every transition calls `govern::check` first. Budgets are *layered* (ch.01 §4.10 config): defaults → profile → per-run override → user steer.

#### 4.3.1 The budget object (Appendix A.5)

```rust
struct Budget {
    // Hard ceilings (Aborted on breach):
    max_steps:        u32,   // total transitions through ACT (default 80)
    max_repairs:      u8,    // per step (default 3)        — bounds the repair loop
    max_replans:      u8,    // per run  (default 4)        — bounds the replan loop
    max_wallclock_ms: u64,   // total run wall time (default 30 min interactive; ∞ overnight)
    max_subagents:    u32,   // concurrent + total fan-out (default 8 = runtime max_batch)
    max_stack_depth:  u8,    // search/subagent nesting (default 5) — bounds tree depth
    max_tool_calls:   u32,   // total effectful calls (default 200)
    max_edits_per_file: u8,  // re-edits of one file before forced replan (default 5) — anti-thrash
    // Soft dials (shape spend, do not abort):
    token_budget_hint: u64,  // advisory; LOCAL so default is generous (K4)
    search_breadth:   u8,    // N for best-of-N / ToT branching (default 1 = ReAct)
    search_depth:     u8,    // ToT/LATS depth (default 0)
    self_consistency_k: u8,  // vote size (default 1)
    escalation:       EscalationPolicy,   // §4.8 — when to spend more
}

struct BudgetLedger {       // consumed-so-far, checked against Budget
    steps: u32, repairs: BTreeMap<StepId,u8>, replans: u8,
    wall_started: Instant, tool_calls: u32, edits: BTreeMap<PathBuf,u8>,
    tokens_est: u64, subagents_live: u32, subagents_total: u32, stack_depth: u8,
}
```

**The defining choice (K4): default budgets cap *wallclock, steps, and effect-counts* — NOT token spend.** Because compute is free locally, the natural limiting reagent is *time* and *number of irreversible effects*, not tokens. `token_budget_hint` exists only to inform the Context Compiler and to flag pathological runs; it never aborts by default. An *overnight* profile sets `max_wallclock_ms = ∞`, `max_steps` high, `search_breadth`/`self_consistency_k` large — and lets the machine grind. **This is the budget design cloud agents cannot have: their governor is dominated by dollars-per-token.**

#### 4.3.2 Enforcement & telemetry

```rust
fn check(state: &AgentState, env: &Env) -> Result<()> {
    let (b, l) = (&state.plan.budget, &state.ledger);
    ensure!(l.steps < b.max_steps,            Abort::Steps);
    ensure!(l.replans <= b.max_replans,       Abort::Replans);
    ensure!(elapsed(l) < b.max_wallclock_ms,  Abort::Wallclock);
    ensure!(l.tool_calls < b.max_tool_calls,  Abort::ToolCalls);
    ensure!(l.stack_depth <= b.max_stack_depth, Abort::Depth);
    // per-step / per-file checks happen at their transitions
    Ok(())
}
```

The Governor also *ingests telemetry*: it polls `hawking-serve /metrics` (and reads `GenStats.dec_tps`, `spec_gov` accept rates) to (a) detect a degraded runtime → trigger ch.01's `RuntimeSupervisor` pause, and (b) feed the **escalation controller** (§4.8): if the model is fast (high `dec_tps`) and cheap (local), the controller is *biased to escalate breadth/depth* because the marginal cost is near-zero. Every budget transition emits `budget.transition{kind, consumed, cap}` so the timeline shows *why* a run stopped.

#### 4.3.3 Autonomy levels (the approval gate)

A per-profile `Autonomy` (ch.01 §4.10) governs which effects require human approval, mapped onto the `Paused` state:

| Level | Reads (`fs.read`, `search`) | Writes (`fs.write`, `diff.apply`) | Shell / destructive | Plan |
|---|---|---|---|---|
| `suggest-only` | auto | **propose only** → `Paused` for approval | `Paused` | shown, editable, runs only on approve |
| `auto-apply-with-tests` | auto | auto **iff** verification (build+test) passes | confirm destructive only | auto |
| `autonomous` (overnight) | auto | auto | auto within sandbox (worktree) | auto |

Approval is itself an event (`approval.requested`/`granted`/`denied`), so an overnight run started `autonomous` and a careful refactor started `suggest-only` are both fully replayable, and the user can *change autonomy mid-run* (a steer, §4.4).

---

### 4.4 Abort / interrupt / steer semantics

Three distinct user controls, all routed as **intents** (ch.01 Wire A) and all checked at transition boundaries (§4.2.3):

1. **Abort** (hard cancel). The intent flips the run's `Arc<AtomicBool>` that is plumbed into `GenerateRequest.abort` (a real field in `engine.rs`) — so an in-flight generation bails at the next token boundary with `StopReason::Aborted`. The driver, seeing `interrupt.poll() == Abort`, finalizes any in-flight `tool.result` as `cancelled`, emits `run.aborted`, **checkpoints state** (§4.13 — so the user can resume later), and transitions to `Aborted`. Bounded latency: one token + one transition.
2. **Pause** (suspend). Transitions to `Paused` *after* the current transition completes (never mid-effect). The session worker yields; state is durable. `Resume` re-enters from the saved frame. Pause is also how `suggest-only` waits for approval.
3. **Steer** (inject guidance without losing progress). The user adds a message ("actually, keep the old API") mid-run. It is recorded as `turn.user{steer=true, target_run}`, folded into the **plan** at the next `SELECT_STEP`/`REPLAN` boundary (not the middle of a tool call), and added to the next step's context as a high-priority, pinned span (ch.04 §4.2). The agent does **not** discard completed, verified steps — steer revises the *remaining* DAG. This is the "course-correct without restarting" capability that long-horizon work demands.

**Per-token watchdog.** `GenerateRequest.max_stall_ms` (a real field) is set from the profile so a hung forward step self-aborts — a kill-switch for a wedged runtime, independent of user action.

**Crash interrupt.** A host/runtime crash is the same as abort-without-checkpoint: on restart, the kernel replays the durable log to the last transition (the last `Action` with no `Observation` is the interrupted effect) and offers *resume-forward* (§4.13, §6).

---

### 4.5 Plan-as-data (HTN + dependency DAG)

The plan is the agent's intent **as a typed object**, not prose in a transcript (K3). It is a **Hierarchical Task Network**: a tree of steps (decomposition) whose leaves form a **dependency DAG** (execution order). It is an event stream (`plan.step`), so it is diffable, user-editable, replan-able, and rendered live by ch.03.

#### 4.5.1 The plan schema (normative — Appendix A.1)

```jsonc
{
  "plan_id": "plan_01H…",            // ULID
  "run_id": "run_…",
  "goal": "Add JWT refresh-token support to the auth service",
  "budget": { /* §4.3 Budget */ },
  "status": "active",                // draft|active|done|failed|abandoned
  "steps": [
    {
      "step_id": "s1",
      "parent": null,                 // HTN decomposition tree edge
      "title": "Locate the auth token issuance code",
      "kind": "investigate",          // see StepKind below
      "rationale": "Need the current sign() call site before adding refresh",
      "deps": [],                     // DAG edges (must be done before this runs)
      "status": "done",               // pending|active|done|failed|skipped|blocked
      "acceptance": {                 // the VERIFIER's contract for this step (§4.6)
        "oracles": ["grep_ast"],
        "predicate": "found function `issue_token` in auth/*.rs"
      },
      "produced": ["symbol:issue_token@auth/jwt.rs:42"],   // outputs other steps consume
      "attempts": 1, "repairs": 0,
      "est_tokens": 1200, "actual_tokens": 980
    },
    {
      "step_id": "s2",
      "parent": null,
      "title": "Implement refresh-token issuance + rotation",
      "kind": "edit",
      "deps": ["s1"],
      "status": "pending",
      "acceptance": {
        "oracles": ["patch_apply", "typecheck", "build", "test"],
        "predicate": "cargo build ok ∧ `auth::refresh` tests pass",
        "tests": ["auth::refresh::*"]
      },
      "search_hint": { "strategy": "best_of_n", "n": 4 }   // §4.8 per-step override
    }
    // … s3 add route, s4 update docs, s5 integration test …
  ]
}
```

```rust
enum StepKind {
    Investigate,   // read/search only; no effects; oracle = "found X" (grep/AST)
    Edit,          // propose+apply a diff; oracle = patch_apply+build+test
    Command,       // run a shell/test/build command; oracle = exit code + parsed output
    Verify,        // a pure verification step (run the suite); oracle = test.status
    Synthesize,    // produce an artifact (a summary, a design); oracle = LLM-judge/rubric
    Decompose,     // expand into sub-steps (HTN recursion); oracle = children well-formed
    Delegate,      // hand to a subagent (§4.10); oracle = subagent return contract
}
```

**Every step declares its `acceptance` oracle up front.** This is the most important field: *the plan commits, before acting, to how each step will be verified.* It makes verification non-optional (K1) and turns the abstract goal into a checklist of machine-checkable predicates — the single biggest lever against the "verification gap" that MAST found is 21% of failures.

#### 4.5.2 The DAG & ready-set

Leaf steps form a DAG over `deps`. `SELECT_STEP` computes the **ready set** = `{ s : s.status==pending ∧ ∀ d∈s.deps, d.status==done }` and picks one (deterministic tie-break on `step_id`; or parallelize independent ready steps via subagents, §4.10). Cycle detection runs at `Plan`/`Replan` time (`dag.acyclic()` guard); a cyclic plan is rejected and re-synthesized. Independent branches of the DAG are the natural unit of **parallelism** (K9): N ready steps with disjoint file footprints → N worktree-isolated subagents.

#### 4.5.3 Replanning

Two scopes, chosen by `replan::revise`:

- **Localized replan** (cheap, preferred — the Plan-and-Act lesson): only the *active subtree* is revised; completed/verified steps are untouched; their outputs (`produced`) are preserved. Triggered when a step fails after exhausting repairs but the *overall* plan is still sound (e.g. "the edit approach was wrong, try a different edit" → replace `s2`'s subtree). Bounded by `max_replans`.
- **Full replan**: re-synthesize from the goal, carrying forward all `repair.lesson`s as context (so the new plan doesn't repeat mistakes). Triggered when the goal itself was mis-decomposed (the structure is wrong, not just one step). Returns to `Plan`.

Replanning **always carries lessons forward** (§4.7) and emits `plan.replanned{scope, reason, lessons[]}` so the timeline shows the pivot. A plan that exhausts `max_replans` does not loop forever (K8) — it `Finalize`s with an honest partial report.

---

### 4.6 Verification: the reliability core

This is the chapter's thesis made concrete (K1). **Verification ranks oracles strictly: deterministic > consistency/PRM > LLM-judge.** A model score never overrides a deterministic oracle; it only *breaks ties among oracle-passing candidates*.

#### 4.6.1 The Oracle interface (normative — Appendix A.2)

```rust
/// A verifier of a step's outcome. Deterministic oracles RUN something real
/// (compiler, test, patch-apply) and return a ground-truth Verdict. Probabilistic
/// oracles (LLM-judge, PRM) return a SCORE that only ranks within oracle-passing sets.
trait Oracle {
    fn id(&self) -> &str;                       // "build", "test", "typecheck", "grep_ast", "llm_judge"
    fn class(&self) -> OracleClass;             // Deterministic | Probabilistic
    fn cost_hint(&self) -> Cost;                // cheap (grep) … expensive (full test suite)
    /// Run against the candidate outcome. Deterministic oracles must be PURE w.r.t.
    /// the workspace snapshot (run in a sandbox/worktree, §4.10) so they don't mutate
    /// shared state and so the result is recordable & replayable (K5).
    fn check(&self, c: &Candidate, env: &VerifyEnv) -> Verdict;
}

enum OracleClass { Deterministic, Probabilistic }

struct Verdict {
    oracle: String,
    outcome: Outcome,                 // Pass | Fail | Inconclusive
    score: Option<f32>,               // probabilistic only, ∈ [0,1]
    failures: Vec<Failure>,           // structured: {file, line, code, message, category}
    artifacts: Vec<BytesRef>,         // logs/diffs (content-addressed, ch.01 blob store)
    duration_ms: u64,
}
```

#### 4.6.2 The deterministic oracle suite (the workhorses)

| Oracle | Runs | Pass iff | Failure detail | Cost |
|---|---|---|---|---|
| `patch_apply` | apply the proposed diff to a worktree | hunks apply cleanly, no fuzz beyond ε | rejected hunks, offsets | cheap |
| `typecheck` | language type checker (`tsc --noEmit`, `cargo check`, `mypy`) | exit 0, no type errors | `{file,line,code}` per error | med |
| `build` | the project build (`cargo build`, `npm run build`) | exit 0 | parsed compiler diagnostics | med–high |
| `test` | the relevant test subset (from `acceptance.tests`) | all pass | per-test pass/fail, assertion msgs | high |
| `lint` | linter/formatter (`clippy`, `eslint`, `ruff`) | no errors (warnings configurable) | rule violations | cheap |
| `grep_ast` | structural search (tree-sitter/SCIP, ch.01 §3) | predicate matches (`symbol exists`, `no TODO left`) | matches/misses | cheap |
| `schema` | validate JSON/config artifacts against a schema | valid | path-level errors | cheap |
| `runtime_smoke` | run the changed binary/endpoint with a canned input | expected output/no crash | stdout/stderr diff | high |

**These are the reliability engine.** A 7B model's *proposal* is fallible; `cargo build` is not. The loop advances a step only when its declared `acceptance.oracles` all `Pass`. This is exactly the "tool-integrated self-verification lifts small models" result ([Kang et al. 2025]) — small models become reliable when the *tool*, not the model, is the judge of correctness.

#### 4.6.3 Probabilistic oracles (fallback & tie-break only)

- **Self-consistency / vote** (`consistency.rs`): when a step has no deterministic oracle (e.g. `Synthesize` a design summary), sample K outcomes and take the majority/centroid (Self-Consistency, §3.1). Cheap, local, surprisingly strong.
- **PRM-style stepwise scoring**: score the *trajectory* (not just the final patch) — penalize a candidate that took a suspicious path even if it happens to pass, to break ties toward the cleaner solution. (Generative-verifier prompt; or a small trained PRM later — §8.)
- **LLM-as-judge** (`llm_judge.rs`): the model critiques the candidate against the step's `acceptance.predicate` and a rubric. **Strictly fallback** — used only when `class==Inconclusive` (no deterministic oracle applies) and as a tie-break. Its known biases ([§3.2]) mean it *never* overrides `build`/`test`. We mitigate bias with: pairwise comparison over absolute scoring, position-swapped double-judging, and rubric-grounding (agentic-rubric verifier, §3.2).

#### 4.6.4 The Verification Gate (pseudocode)

```rust
/// Decide a step's fate. Deterministic oracles are AUTHORITATIVE; probabilistic
/// ones only rank within the deterministic-pass set (the verifier-quality ceiling, §3.2).
fn verify(state, env, mode) -> StepOutcome {
    let step = state.plan.step(state.cursor);
    let cand = state.current_candidate();           // the step's outcome (diff/output)
    let det: Vec<Verdict> = step.acceptance.oracles.iter()
        .filter(|o| o.class()==Deterministic)
        .map(|o| o.check(cand, env)).collect();     // RUN them (sandboxed)
    emit_all(det.iter().map(|v| event("verify.result", v)));

    if det.iter().any(|v| v.outcome == Fail) {
        let failures = collect_failures(&det);      // structured, minimal (§4.7)
        return decide_repair_or_replan(state, failures);
    }
    if det.iter().all(|v| v.outcome == Pass) && !det.is_empty() {
        commit_step(state, cand);                   // e.g. diff.applied (the ONLY effect-commit)
        return advance(state);                      // → SELECT_STEP (done)
    }
    // No deterministic oracle applied → fall back to probabilistic (gated)
    let score = probabilistic_score(cand, step, env);   // consistency vote ⊕ judge
    if score >= step.acceptance.threshold.unwrap_or(0.7) { commit_step(state,cand); advance(state) }
    else { decide_repair_or_replan(state, judge_failures(cand, step)) }
}
```

**Effect-commit happens only inside the gate, only on Pass.** A `diff.applied` (a real file write) is emitted *after* `build`+`test` pass on a worktree copy — so the working tree is never left broken. This is the structural guarantee against the "agent breaks the build and walks away" failure: the build *is* the gate.

#### 4.6.5 Confidence-gated verification depth **[RUNTIME-SIDE — LATER]** / [SHELL-TODAY] fallback

We can spend *less* verification when the model is confident and *more* when it is not (a local superpower, §4.12): read the generation's logprobs/entropy (a `logprobs` field on the SSE token event — the LATER hook) and scale the oracle suite — a high-confidence, low-entropy edit runs `typecheck`+`test`; a low-confidence, high-entropy one *also* runs `runtime_smoke` and triggers best-of-N. **[SHELL-TODAY] fallback**: without logprobs, gate on *outcome proxies* — if `patch_apply` needed fuzz, or the first oracle failed once, escalate depth. Either way the dial is the same: confidence ↔ verification depth.

---

### 4.7 Self-correction (Reflexion-style minimal repair)

When `VERIFY` fails with repair budget left, the loop enters `REPAIR` (K6). The discipline: **the repair prompt contains the *smallest* high-signal failure context, not the whole history.**

#### 4.7.1 The minimal repair context

```rust
struct RepairContext {
    step: StepRef,                    // what we were trying to do (title + acceptance)
    last_attempt: DiffOrOutput,       // exactly what we produced (not the reasoning that led to it)
    failures: Vec<Failure>,           // the STRUCTURED oracle failures (the build errors, the
                                      //   failing assertions) — verbatim, deduped, capped
    lesson: Option<LessonRef>,        // prior reflection on THIS class of failure (if any)
    minimal_code: Vec<SpanRef>,       // only the spans the failures point at (ch.04 realize())
}
```

This is deliberately *not* the full transcript. The model gets: *"You tried this diff; the compiler said these 3 errors at these lines; here are those lines; fix it."* Reflexion showed verbal self-reflection on *failures specifically* beats re-running with the whole context, and ch.04's context-rot evidence says the whole context actively *hurts*. The repair context is reconstructed each attempt from structured failures, so it never accumulates cruft.

#### 4.7.2 The reflection → lesson pipeline

```
VERIFY fail ──▶ summarize the failure into a "lesson" (1–3 sentences, the Reflexion artifact):
                "When editing auth/jwt.rs, the `Claims` struct requires `exp` as i64 not u64;
                 the build fails with E0308 otherwise."
            ──▶ store as episodic memory (ch.04 MemoryStore, type=episodic, keyed by step+repo)
            ──▶ prepend to the NEXT repair attempt's RepairContext.lesson
            ──▶ on Finalize, promote durable lessons → semantic/procedural (skill, §4.11)
```

Lessons are scoped (`repo + symbol + failure-category`) and retrieved by the same recency/importance/relevance scorer ch.04 uses, so a failure that recurs across sessions is recalled *before* it is repeated. A lesson that proves a *general* recipe ("to add a route in this repo, register it in `router.rs::build`") is promoted to a **skill** (§4.11). This is the loop's *learning* mechanism — it gets better at *this repo* without any weight update.

#### 4.7.3 Repair vs replan boundary

`decide_repair_or_replan` (the gate's tail):

- **Repair** (same step, new attempt) when the *approach* is sound but the *execution* was wrong (fixable from the failure detail). Bounded `max_repairs` per step.
- **Replan** when repeated repairs fail (the approach is wrong) or the failure reveals the *plan* was wrong (a missing dependency, a wrong decomposition). Localized first, full if needed.
- A step that exhausts repairs *and* triggers a localized replan that also fails → the step is marked `failed` and the run `Finalize`s honestly (no infinite repair, K8).

---

### 4.8 Search & sampling-scale strategies

This is where **free local compute becomes reliability** (K4). All strategies nest *inside* `ACT` as a sub-machine on the `stack`; the **escalation ladder** decides which to use per step. The default is the cheapest (ReAct, breadth 1); we climb only when the step is hard or the first attempt fails.

#### 4.8.1 The strategy ladder (cheapest → most expensive)

| Tier | Strategy | When | Compute | Local advantage |
|---|---|---|---|---|
| **0** | **ReAct (single trajectory)** | default; trivial/well-specified steps | 1× | baseline |
| **1** | **Self-consistency (vote)** | step has no deterministic oracle; ambiguous output | N× parallel, vote | N parallel decodes are *free* on a local box; pick majority |
| **2** | **Best-of-N + oracle** | hard `edit`/`command` steps; oracle can rank | N× parallel → verify each → keep oracle-passing best | **the killer app**: N candidate diffs, `build`+`test` each, keep the one that passes — cloud rations N, we don't |
| **3** | **Tree-of-Thoughts** | branchy decisions (design choice, multi-approach) | N^depth, pruned by eval | explore approaches in parallel, prune by oracle/self-eval |
| **4** | **LATS / MCTS over tool trajectories** | long, uncertain, high-value steps where *sequencing* matters | many rollouts, UCT-guided, backprop reward | full tree search over `(reason,act,observe)`; reward = oracle pass + LM value; reflection on failed rollouts — only viable *because* compute is free |
| **5** | **Debate / critic panel** | rare; genuinely contested correctness with no clean oracle | K agents + judge | §4.14 — gated; usually *not* worth it |

#### 4.8.2 Best-of-N + oracle (the workhorse — pseudocode)

```rust
/// Tier 2. Generate N candidate outcomes for a step IN PARALLEL (isolated worktrees,
/// K9), verify each with the step's deterministic oracles, keep the oracle-passing
/// candidate with the best tie-break score. This is the single highest-value use of
/// free local compute: it converts a weak model's "maybe-right" into a verified
/// "definitely-passes-the-build".
fn best_of_n(step, env, n) -> Candidate {
    let cands: Vec<Candidate> = (0..n).into_par_iter()        // N parallel session workers
        .map(|i| {
            let seed = base_seed + i;                         // DISTINCT seeds → diverse, REPRODUCIBLE
            let wt   = env.fork_worktree();                   // isolation (K9)
            generate_candidate(step, env.with(seed, wt))      // ReAct in the fork
        }).collect();
    let verified: Vec<(Candidate, Verdict)> = cands.into_par_iter()
        .map(|c| (c.clone(), run_oracles(&c, step)))          // build+test each, in its worktree
        .filter(|(_, v)| v.outcome == Pass)                   // ORACLE-GATED (K1)
        .collect();
    match verified.into_iter().max_by_key(|(c,_)| tie_break_score(c)) {  // §4.8.4
        Some((best, _)) => { merge_worktree(best); best }     // adopt the winner's edits
        None            => repair_or_escalate(step, env),     // none passed → REPAIR or climb a tier
    }
}
```

**Why this is the centerpiece.** It is the cleanest expression of the thesis: *spend free parallel compute to generate variety, let deterministic oracles select truth.* N=4–8 diffs, each built and tested in an isolated worktree, keep the passing one. A cloud agent at $/token avoids this; we default to it on hard steps. Distinct seeds give *diverse* candidates that are still *reproducible* (K2) — the run replays exactly.

#### 4.8.3 LATS over tool trajectories (the deep tier)

For long, uncertain steps, `lats.rs` runs MCTS where a node is `(reasoning, action, observation)` and:
- **Selection**: UCT over children (`value + c·√(ln N_parent / N_child)`).
- **Expansion**: sample K candidate actions (tool calls) under constraint (§4.9).
- **Simulation/eval**: run the action's oracle (deterministic reward) + an LM self-evaluation of partial progress (value head).
- **Backprop**: propagate reward up the path; **reflection** on failed rollouts is stored as a lesson (§4.7) so sibling branches don't repeat the mistake.
- **Budgeted**: `max_stack_depth` and a rollout cap from the Governor bound the tree.

This is gated to *high-value* steps (a tricky multi-file refactor, a flaky-test root-cause) because even free compute has a wallclock cost. Its reward signal is *our deterministic oracles*, which is exactly why LATS-for-code works here: the environment reward is a real build/test, not a guess.

#### 4.8.4 Candidate scoring (oracle-first, model-second)

```
score(c) =  𝟙[all deterministic oracles pass]          · BIG        # GATE — non-passing → -∞
          + w_test   · (tests_passed / tests_total)                  # partial credit among passers
          + w_clean  · cleanliness(c.diff)                           # smaller, localized diffs win
          + w_prm    · prm_step_score(c.trajectory)                  # PRM/consistency, tie-break only
          − w_risk   · risk(c)                                       # touched-files, destructive ops
          − w_tokens · c.tokens · ε                                  # tiny: prefer concise (local→ε~0)
```

The `𝟙[oracles pass]` term *dominates* — model-based terms only order the survivors. This bakes the verifier-quality ceiling (§3.2) into the math: we never select an oracle-failing candidate because a model "liked" it.

#### 4.8.5 The escalation controller

```rust
/// Decide the tier for a step. Climbs on difficulty signals; biased to climb HARDER
/// when the runtime is fast/cheap (free compute, K4). Records the choice as an event.
fn pick_tier(step, env, ledger) -> Tier {
    let mut tier = step.search_hint.tier.unwrap_or(Tier::React);     // plan may pre-hint
    if step.kind == Edit && step.acceptance.has_deterministic() { tier = max(tier, Tier::BestOfN); }
    if step.prior_failures() >= 1 { tier = tier.next(); }            // failed once → spend more
    if step.is_branchy_decision() { tier = max(tier, Tier::ToT); }
    if step.is_high_value_uncertain() { tier = max(tier, Tier::Lats); }
    if env.runtime_is_fast() && env.is_overnight() { tier = tier.bump(); }  // lavish (K4)
    clamp(tier, env.budget.max_tier_for(step))
}
```

The point the brief insists on: **the decision of *how much to spend* is itself a first-class, recorded, budget-bounded policy** — not an afterthought. Overnight, the controller is generous; interactively, it stays cheap until a step proves hard.

---

### 4.9 The tool-call protocol

Tools are how the agent *acts*. The protocol is the contract between the model's textual output and real effects. It must be (a) **valid by construction** (constrained decode), (b) **capability-checked** (ch.01 T4), (c) **idempotent & replay-safe** (K5), and (d) **linted at the boundary** (the SWE-agent ACI lesson).

#### 4.9.1 The envelopes (normative — Appendix A.3)

```jsonc
// ToolCall — an Action-class event (ch.01 `tool.call`)
{
  "call_id": "tc_01H…",               // ULID; idempotency key
  "tool": "fs.write",                  // registry id (ch.01 §7 tool registry)
  "args": { "path": "auth/jwt.rs", "diff": "@@ … @@" },   // VALIDATED against tool's arg schema
  "capability_grant_id": "grant_…",    // the scoped grant authorizing this (ch.01 T4)
  "idempotency_key": "blake3(tool+canonical_args)",       // dedup identical calls
  "expects": "diff_applied | text | exit_code | json",    // declared result shape
  "timeout_ms": 30000,
  "dry_run": false                     // suggest-only renders the call without effect
}
// ToolResult — an Observation-class event (ch.01 `tool.result`), cause = call_id
{
  "call_id": "tc_01H…",
  "ok": true,
  "output": { /* shape per `expects` */ },
  "bytes_ref": "blake3:…",             // large outputs → blob store (ch.01 §4.7)
  "exit_code": 0,
  "duration_ms": 412,
  "side_effects": ["wrote auth/jwt.rs (+18 -2)"],          // for the timeline + undo
  "error": null                         // structured error if !ok (taxonomy code, §6)
}
```

#### 4.9.2 Emission under constraint (valid-by-construction) **[RUNTIME-SIDE — LATER]** / [SHELL-TODAY]

The model emits tool calls as **constrained JSON** so a malformed call is *architecturally impossible* (§3.6). The tool's argument schema is compiled to a grammar; the decode is masked via the in-tree `JsonConstraint::mask_logits` (today: JSON-mode; the LATER extension: full JSON-Schema/grammar with tag-triggered tool-name→args switching, XGrammar-style). **[SHELL-TODAY] fallback**: `json_mode=true` on the request gives structural JSON validity now; a thin post-parse validator + one constrained re-prompt on schema mismatch closes the rest. Either way, the dispatcher *never* sees an unparseable call.

```rust
fn emit_tool_call(step, env) -> ToolCall {
    let tools = env.available_tools(step);                 // capability-filtered set
    let grammar = compile_toolcall_grammar(&tools);        // tool-name → that tool's arg schema
    let raw = env.runtime.generate_constrained(prompt(step), grammar);  // mask_logits (§4.12)
    let call = parse_toolcall(raw);                        // cannot fail structurally under constraint
    lint::validate(&call, &tools)?;                        // SEMANTIC checks below (§4.9.3)
    call
}
```

#### 4.9.3 Pre-flight lint (the ACI guardrail)

Before dispatch, `tools/lint.rs` rejects bad calls *with a corrective message* (re-prompt, don't fail the run):
- **Unknown tool** / tool not in the granted set → reject + list available.
- **Arg-schema violation** (despite constraint, defense-in-depth) → reject + schema.
- **Hallucinated file** (path doesn't exist for a read/edit) → reject + nearest real paths (anti-hallucination, §6). *This single check kills a whole failure class.*
- **Syntactically-broken edit** (the diff would produce un-parseable code — checked with tree-sitter before apply) → reject + the parse error. (SWE-agent's edit-linter, generalized.)
- **Out-of-scope path** (write outside workspace, or to a denied path) → reject (capability, T4).

#### 4.9.4 Dispatch, idempotency & replay

```rust
fn dispatch(call: ToolCall, env, mode) -> ToolResult {
    if mode == Replay { return env.recorded_result(call.call_id); }   // K5: replay folds outcome
    if let Some(prev) = env.idempotency_cache.get(&call.idempotency_key) {
        return prev;                                                  // dedup identical calls
    }
    env.capability_check(&call)?;                                     // T4 — deny beats allow
    if env.autonomy.requires_approval(&call) { return pause_for_approval(call); }  // §4.3.3
    let result = env.registry.invoke(&call);                         // the real effect
    emit("tool.result", &result);                                    // Observation, cause=call_id
    env.idempotency_cache.put(call.idempotency_key, result.clone());
    result
}
```

**Idempotency** prevents the "agent re-runs `npm install` five times" thrash and makes retries safe. **Replay short-circuit** is the K5 guarantee: during replay/scrub, `dispatch` returns the *recorded* `ToolResult` and never re-fires the effect. **Capability binding** ties every call to a scoped grant (ch.01 T4) so a tool can only do what was explicitly authorized.

#### 4.9.5 Built-in tool families (the action space)

The kernel ships a *minimal, LM-friendly* action space (SWE-agent ACI lesson — fewer, well-shaped tools beat many sharp ones). All are extensions (ch.01 §7); these are the defaults:

`fs.read` · `fs.write`(diff-based) · `fs.list` · `search.grep` · `search.symbol`(SCIP) · `shell.run`(sandboxed) · `test.run` · `build.run` · `git.*`(read-mostly) · `plan.update`(the agent edits its own plan) · `memory.write` · `subagent.spawn` · `skill.invoke`. Each is small, schema'd, and lintable. The **`plan.update`** tool is notable: the agent revises its plan-as-data through the same audited protocol as any other effect.

---

### 4.10 Subagents: spawn / delegate / return / isolate

Subagents are how the kernel **parallelizes** (independent DAG branches, best-of-N candidates) and **isolates context** (a clean window for a sub-task, returning a tight summary — the Anthropic sub-agent pattern, ch.04 §3.5). The default is *single-agent*; subagents are spawned only when work is **separable and independently verifiable** (K9, K12).

#### 4.10.1 The delegation contract (normative — Appendix A.4)

```jsonc
// SubagentSpec — the parent's request
{
  "subagent_id": "sa_01H…",
  "parent_run": "run_…",
  "goal": "Find every call site of `issue_token` and report file:line + signature",
  "kind": "research",                  // research | implement | verify | review
  "isolation": "worktree",             // none | context | worktree (K9)
  "budget": { "max_steps": 20, "max_wallclock_ms": 120000, "max_tier": "best_of_n" },
  "context_seed": ["symbol:issue_token", "file:auth/jwt.rs"],   // minimal handoff (ch.04)
  "return_contract": {                 // EXACTLY what the parent expects back
    "shape": "summary",                // summary | diff | verdict | artifact_ref
    "schema": { "call_sites": "[{file,line,signature}]" },
    "max_tokens": 1500                 // the return is SMALL (clean-window discipline)
  },
  "deadline": "2026-06-24T…"
}
// SubagentReturn — the child's reply (an Observation to the parent)
{
  "subagent_id": "sa_01H…",
  "status": "ok",                      // ok | partial | failed | aborted
  "result": { "call_sites": [ /* … */ ] },   // conforms to return_contract.schema
  "summary": "Found 3 call sites …",   // ≤ max_tokens — the ONLY thing entering parent context
  "lessons": ["…"],                    // promoted to parent's episodic memory
  "budget_used": { /* ledger */ },
  "artifacts": ["blake3:…"]            // full detail in blob store; NOT inlined to parent
}
```

#### 4.10.2 Isolation levels

- **`none`** — shares the parent's context (cheap, for a quick sub-question). Rare.
- **`context`** — fresh, minimal context window (the `context_seed` only); shares the workspace. The Anthropic clean-window pattern: the subagent burns its own tokens reading files and returns a 1–2k-token summary, keeping the parent's context clean. Default for `research`/`review`.
- **`worktree`** — a git worktree (or overlay FS) clone; effects are isolated until the parent merges the verified result. **Default for `implement`** and for *every* parallel attempt (best-of-N, parallel DAG branches). This is how N agents edit "the same file" without colliding (K9) — each in its own tree, oracle-verified, then merged.

#### 4.10.3 The spawn/return mechanics

```rust
fn spawn(spec: SubagentSpec, env) -> SubagentHandle {
    env.governor.charge_subagent(&spec)?;                  // counts against max_subagents (K8)
    let iso = env.isolate(spec.isolation);                 // fork worktree / fresh context
    let child = AgentKernel::new_run(spec.goal, spec.budget, iso);  // a NESTED state machine
    emit("subagent.spawned", &spec);
    env.registry.track(child)                              // for cancellation, budget rollup
}
fn join(handle, env) -> SubagentReturn {
    let ret = handle.await_return();                       // child runs its own loop to terminal
    emit("subagent.returned", &ret);
    fold_into_parent(env, &ret);                           // ONLY the summary+result enter context;
                                                           // lessons → episodic; artifacts → blob ref
    if spec.kind == Implement && ret.status == Ok {
        merge_worktree_after_verify(env, &ret);            // oracle-gated merge (K1)
    }
    ret
}
```

A subagent **is just a nested `AgentState`** on the stack — same state machine, same governor (with the child budget), same event envelope (its events carry the child `run_id` and `parent` = the spawning step). This recursion is bounded by `max_stack_depth` (K8). The parent only ever ingests the *summary* — never the child's raw transcript — which is the discipline that keeps long multi-agent runs from drowning in context (the long-horizon-SWE lesson, §3.3).

#### 4.10.4 Cancellation & failure propagation

Aborting the parent cascades `abort` to all live children (each flips its `GenerateRequest.abort`). A child that `failed` returns a structured failure; the parent decides repair/replan as if a step failed (§4.7). A child that exhausts its budget returns `partial` — the parent never blocks forever on a subagent (K8).

---

### 4.11 The Skill Library (Voyager-style, persistent)

The kernel **gets better at *this* repo over time** by capturing verified solutions as reusable skills that survive sessions (K10). This is Voyager applied to coding tools: generate a skill, *validate it by execution* (our oracles), store it indexed by description, retrieve by similarity, reuse.

#### 4.11.1 The Skill schema (normative — Appendix A.6)

```jsonc
{
  "skill_id": "skill_01H…",
  "name": "add_axum_route",
  "description": "Register a new HTTP route in this repo's axum server",  // embedded for retrieval
  "kind": "procedure",                 // procedure | snippet | recipe | macro
  "trigger": "when adding an HTTP endpoint",                  // when to retrieve it
  "body": {                            // executable / templated steps, NOT free prose
    "steps": [
      "add handler fn in src/http/handlers/<name>.rs",
      "register in src/http.rs::build_router via .route(\"<path>\", <method>(<handler>))",
      "add request/response structs with serde derives",
      "add a test in tests/http/<name>.rs"
    ],
    "params": ["name", "path", "method"],
    "example_diff_ref": "blake3:…"     // a worked example from when it was learned
  },
  "provenance": { "learned_from_run": "run_…", "repo": "hawking", "commit": "abc123" },
  "validation": { "last_verified": "…", "oracle": "build+test passed", "success_count": 7,
                  "fail_count": 0 },   // execution-validated, Voyager-style
  "importance": 0.8, "last_access": "…", "access_count": 7,
  "supersedes": "skill_…",             // version chain
  "embedding_ref": "vec:…"
}
```

#### 4.11.2 Capture (on success) → curate → retrieve

```
FINALIZE (a step/run succeeded with non-trivial structure) ──▶
   distill the verified trajectory into a Skill body (the recipe that worked) ──▶
   VALIDATE the skill is reusable (re-run its oracle on the worked example) ──▶
   store in the Skill Library (ch.04 MemoryStore, type=procedural) with embedding ──▶
   [later turns] PLAN/ACT retrieve top-k skills by description-vs-goal similarity ──▶
   inject the skill body into the planner/step context (ch.04 realize()) as a recipe
```

- **Capture-on-success** (Voyager): only *verified* solutions become skills (a skill that didn't pass oracles is just a lesson, §4.7). The skill stores the *recipe* (the generalizable steps), not the one-off diff.
- **Curate** (Memp/contrastive-refinement, §3.4): skills that keep succeeding get `importance` bumped; skills that fail get demoted/retired (`supersedes`); near-duplicate skills are merged. A background "skill gardener" (idle-time, ch.04 §4.7 sleep-time pattern) consolidates the library.
- **Retrieve** with the recency/importance/relevance score (ch.04) so the *right* skill surfaces. A retrieved skill turns a from-scratch reasoning problem into a *fill-in-the-recipe* problem — exactly what makes a small model reliable.

#### 4.11.3 Why this is a moat, not a cache

A cloud agent's "memory" resets per session or lives in a vendor's opaque store. HIDE's Skill Library is **the user's file** (`.hide/memory/procedural/`, ch.04), versioned, greppable, editable, and *compounding*: every verified solution makes the next similar task faster and more reliable. Over weeks on one repo, the agent accumulates a repo-specific procedural memory that no fresh cloud session can match. The skills are validated by *execution against this repo's build*, so they are *true here* in a way a generic model's prior cannot be.

---

### 4.12 Model-cooperation hooks

Because we own the runtime, the kernel treats the model as a **cooperating party** (K7), not a sealed text box. Four hooks, each with a shipping-today path and a deeper LATER path.

#### 4.12.1 Constrained decode for plans & tool-calls **[partly LATER]**
- **What.** Emit the plan JSON and every tool-call under a grammar/schema constraint so they are *valid by construction* (§3.6, §4.9.2).
- **Hook.** `JsonConstraint::mask_logits` (in-tree) → **[SHELL-TODAY]** JSON-mode validity now; **[LATER]** full JSON-Schema/grammar + tag-triggered tool-name→args switching (XGrammar-class) for zero-overhead exact tool grammars.
- **Impact.** Eliminates the entire "malformed plan / unparseable tool-call" failure family (§6). High impact, low difficulty — the substrate exists.

#### 4.12.2 Logit confidence → verification gating **[LATER]** / [SHELL-TODAY] proxy
- **What.** Read per-token logprobs/entropy to estimate the model's *confidence* in a generation, and scale verification depth + search breadth accordingly (§4.6.5, §4.8.5): confident → cheap verify; uncertain → best-of-N + deeper oracles.
- **Hook.** A `logprobs`/`entropy` field on the SSE token event (the LATER seam on `/v1/hawking/generate`). **[SHELL-TODAY] proxy**: outcome-based confidence (did the first oracle pass? did the patch need fuzz?).
- **Impact.** Spends free compute *where it's needed* instead of uniformly — the efficiency lever for the escalation ladder.

#### 4.12.3 Entropy-triggered escalation **[LATER]**
- **What.** A *spike* in next-token entropy at a decision point (the model is unsure which approach to take) auto-triggers Tier-3 ToT branching at exactly that fork — branch where the model is uncertain, commit where it's confident.
- **Hook.** Same logprobs stream, watched live.
- **Impact.** Targets the most valuable search exactly at the model's points of doubt — a uniquely-local capability (cloud doesn't stream you the entropy to act on mid-decode).

#### 4.12.4 Speculative self-drafting **[SHELL-TODAY via existing SpeculateMode]**
- **What.** The runtime already has `SpeculateMode::{ExactShared, Eagle5}` with **greedy bit-identity at temp=0** and a `spec_gov` accept-rate governor. The kernel *uses* this for raw decode speed (more tokens/sec → more attempts per wallclock → the escalation ladder can afford to climb).
- **Hook.** Set `speculate` on the engine config / request; read `GenStats.{draft_accepted,draft_rejected}` and `spec_gov` metrics into the Governor's telemetry.
- **Future (`fine-tune-at-Condense`, §8).** *Hawking Condense* can fine-tune the served model to natively emit *our* plan/tool-call protocol and to be a better *self-drafter for agent scaffolds* — co-designing the model to the harness. This is the deepest local superpower: the model and the loop are trained together. Marked **[LATER / not shell-gating]**.

---

### 4.13 Checkpoint / replay / resume

Agent state is **durable, deterministic, and resumable across days** (K2, K5). This binds tightly to ch.01's event log (the kernel adds *no* new system-of-record — checkpoints are an optimization over the log) and ch.04's KV-checkpoint hook.

#### 4.13.1 What a checkpoint is

```rust
struct AgentCheckpoint {
    run_id: RunId,
    at_seq: u64,                      // the event-log seq this snapshot folds up to (ch.01)
    state: AgentState,                // §4.2 — small, serializable (plan, cursor, ledger, stack)
    context_manifest: ManifestRef,    // ch.04 — exactly what the model saw (for byte-exact resume)
    kv_checkpoint: Option<KvCheckpointId>,  // ch.04 §4.5.5 — warm-resume the runtime KV [LATER]
    seeds: SeedState,                 // base_seed + per-attempt offsets → deterministic replay (K2)
}
```

A checkpoint is taken **after any transition** (`checkpoint::maybe` — cheap, the state is small) and *always* before `Aborted`/`Paused`. It references the log seq, the context manifest, and (LATER) the KV snapshot — so resume is *exact*, not approximate.

#### 4.13.2 The three operations (one mechanism)

Per K5, **replay, resume, and fork are the same fold over the log with effects disabled** — they differ only in where new events go:

- **Replay/scrub** (read-only time travel): fold events `(snapshot, S]` through `driver.step(mode=Replay)`. Tool calls and generations return their *recorded* outcomes (K5); no effect re-fires. Drives ch.03's timeline scrubber and "why did the agent do that" inspection.
- **Resume** (continue a run after quit/crash/sleep, possibly days later): restore the nearest `AgentCheckpoint`, fold to head in `Replay` mode (rebuild state), then flip to `Live` and continue `driver.step` forward. The runtime KV is warm-restored (`kv_restore`, ch.04 §4.5.5) **[LATER]** or re-established via the prefix cache **[SHELL-TODAY]** — slower to warm, identical result.
- **Fork** (branch a run — "try a different approach from step 7"): identical to resume, but new events write to a *child* `run_id` (ch.01 `session.forked`), leaving the original intact. This is the substrate for *parallel-universe* exploration (run 4 forks overnight, keep the one whose oracles pass best).

```rust
fn resume(run_id, env) -> AgentState {
    let ckpt = env.checkpoints.nearest(run_id);              // or seq 0 if none
    let mut state = ckpt.state.clone();
    env.replay_seeds(ckpt.seeds);                            // K2 — same seeds → same trajectory
    for ev in env.log.range(ckpt.at_seq.., run_id) {         // fold to head, NO effects (K5)
        driver::step(&mut state, env, Mode::Replay);
    }
    warm_runtime(env, &ckpt);                                // KV restore [LATER] or prefix-cache [TODAY]
    state                                                    // now flip to Live and continue
}
```

#### 4.13.3 Determinism guarantee & its boundary

Given the same log prefix + the same seeds, the trajectory is byte-identical (K2) — because the runtime guarantees greedy bit-identity (`SamplingParams.seed`, temp=0) and every effect is recorded. Non-determinism is quarantined to (a) `temperature>0` generations and (b) external tool results (network) — both *recorded*, so even a non-deterministic *live* run *replays* deterministically. This is what makes "resume a 3-day-old overnight run" and "reproduce the bug the agent hit" the *same, reliable* operation — a guarantee no cloud agent offers.

---

### 4.14 Multi-agent topologies & when single-agent wins

The literature's verdict (§3.5) is the policy: **single-agent is the default; multi-agent must earn its place** (K12). Most multi-agent failure is bad task-spec + missing verification (MAST), both of which our plan-as-data + oracle-first design already solves *single-agent*.

#### 4.14.1 The topologies (and our stance on each)

| Topology | Shape | HIDE stance | When |
|---|---|---|---|
| **Single agent (default)** | one state machine, sequential DAG | **Default.** Lowest coordination risk; strongest under equal compute (§3.5). | almost always |
| **Planner / Executor split** | planner emits plan-as-data; executor runs steps | **Used as roles, not processes.** Our `Plan` vs `Act` phases *are* this split, in one machine. | inherent to the loop (§4.2) |
| **Parallel executors (fan-out)** | N subagents on independent DAG branches / best-of-N | **Yes — the main multi-agent use.** Each isolated (worktree), each oracle-verified, merged on pass (K9). | separable, independently-verifiable work |
| **Critic / reviewer** | a reviewer subagent checks the implementer's diff | **Yes, as a *probabilistic oracle* (§4.6.3), gated below deterministic ones.** A code-review subagent adds signal, never overrides `build`/`test`. | high-stakes diffs, no clean oracle |
| **Researcher** | a clean-window subagent gathers context, returns a summary | **Yes — the clean-window pattern** (§4.10, ch.04 §3.5). | context-heavy investigation |
| **Debate / panel** | K agents argue, a judge decides | **Rarely.** Evidence says it often *hurts* (majority pressure, §3.5). Reserved for genuinely contested correctness with no oracle. | last resort |

#### 4.14.2 The decision rule

```
Use a subagent IFF the work is:
  (1) SEPARABLE      — a self-contained sub-goal with a clear input/output contract, AND
  (2) VERIFIABLE     — the subagent's return can be checked (an oracle, a return schema), AND
  (3) ISOLATABLE     — its effects don't entangle with concurrent work (worktree/clean context), AND
  (4) WORTH IT       — parallelism or context-isolation gain > coordination + merge cost.
Otherwise: a single agent with deeper SEARCH (§4.8) beats more agents (§3.5).
```

The crucial reframing: **HIDE prefers "one agent + more search" over "more agents"** because our search tiers (best-of-N, ToT, LATS) are oracle-grounded and coordination-free, whereas multi-agent adds the exact failure modes MAST catalogued. Subagents are for *parallelism and context-isolation*, not for *deliberation* — deliberation is better served by single-agent test-time search against deterministic oracles. This is the decisive, final stance on the topology question.

---

## 5. How we EXCEED (local superpowers)

What the local plane makes possible that **cloud literally cannot do** — each tied to a mechanism above.

1. **Best-of-N every hard step, by default (§4.8.2).** No per-token bill, no rate limit → generate 8 candidate diffs, `build`+`test` each in isolated worktrees, keep the one that passes. Cloud agents ration N because each candidate costs dollars; we default to it. *This single capability is the largest reliability gap in our favor.*
2. **Overnight, resumable, autonomous runs (§4.3.1, §4.13).** A budget with `max_wallclock_ms = ∞`, large search breadth/depth, `autonomous` autonomy, grinding on a hard task while the user sleeps — checkpointed every transition, resumable across days, deterministically replayable. A metered cloud agent cannot economically run for 8 hours; we can, for the cost of electricity.
3. **Constrained-decode plans & tool-calls (§4.9.2, §4.12.1).** We mask the model's logits to a grammar so malformed plans/tool-calls are *impossible*, eliminating a whole failure family. Cloud APIs expose only coarse "JSON mode"; we control the sampler.
4. **Logit-confidence & entropy-gated effort (§4.6.5, §4.8.5, §4.12.2–3).** We read the model's raw confidence and *branch search exactly where it's uncertain, verify cheaply where it's sure.* Cloud doesn't stream you actionable per-token entropy mid-decode.
5. **Deterministic replay/resume/fork as one operation (§4.13).** Greedy bit-identity + recorded effects → "resume a 3-day-old run" and "reproduce the agent's bug" are the *same reliable fold*. No cloud agent offers byte-exact session resume.
6. **A compounding, repo-specific Skill Library that is the user's file (§4.11).** Verified solutions become reusable, execution-validated recipes that survive sessions and make the agent monotonically better at *this* repo — owned by the user, not a vendor.
7. **Oracle-first verification with the *real* toolchain in the loop (§4.6).** The build, the type checker, the test suite *are* the judges, run locally with full filesystem access — no sandboxing tax, no upload, no rate limit on how many times we re-verify. Free re-verification is what lets us spend lavishly on search.
8. **Co-designed model (`fine-tune-at-Condense`, §4.12.4, §8).** *Hawking Condense* can train the served model to natively speak our plan/tool protocol and self-draft agent scaffolds. The model and the loop evolve together — the deepest moat, impossible when you rent someone else's frozen model.

---

## 6. Failure taxonomy → recovery matrix

The complete catalogue of how agent loops fail, each mapped to the mechanism that handles it. (Synthesized from MAST, the hallucination-taxonomy survey, TRAIL, and the in-tree run experience.) Tagged by *detector* and *recovery*.

| # | Failure | Symptom | Detector | Recovery (mechanism) | Bound |
|---|---|---|---|---|---|
| **F1** | **Infinite loop** (same action repeated) | identical `tool.call` idempotency keys recur | idempotency cache (§4.9.4) + cycle check | dedup short-circuits; after K repeats → forced `Replan` | `max_steps`, idempotency (K8) |
| **F2** | **Lost goal / drift** | actions stop serving the plan | plan-step alignment check at `SELECT_STEP`; goal-vs-action relevance | re-pin goal to context head (ch.04); steer; `Replan` | `max_replans` |
| **F3** | **Invalid tool-call** | malformed/unschema'd call | constrained decode (impossible) + pre-flight lint (§4.9.3) | corrective re-prompt under constraint | `max_repairs` then `Replan` |
| **F4** | **Hallucinated file/symbol** | edit/read targets a nonexistent path | lint: path-exists + symbol-exists (SCIP) (§4.9.3) | reject + offer nearest real paths/symbols | re-prompt; bounded |
| **F5** | **Broken build left behind** | edit compiles in the model's head, not reality | `build`/`typecheck` oracle *before* commit (§4.6.4) | effect-commit gated on Pass → worktree never merged broken | structural (cannot happen) |
| **F6** | **Flaky test** | non-deterministic pass/fail | re-run N times; quarantine if unstable; consult flaky-history memory | mark flaky, don't trust as oracle, surface to user | bounded re-runs |
| **F7** | **Infinite edits / thrash** | same file edited over and over | `max_edits_per_file` ledger (§4.3.1) | forced `Replan` after the cap; capture a lesson | `max_edits_per_file` (K8) |
| **F8** | **Context overflow** | window exceeds budget | Context Compiler budget (ch.04 §4.2) | compaction / subagent isolation / demote-not-drop (ch.04) | compiler invariant |
| **F9** | **Context rot** (degradation as context grows) | quality drops though under limit | redundancy penalty + minimal-repair (ch.04 §6, §4.7) | keep smallest high-signal set; clear tool results | compiler policy |
| **F10** | **Premature "done"** | claims success, goal unmet | `acceptance` predicate + final goal-verification step | `Verify` rejects; not `Done` until oracle-confirmed (K1) | gate (structural) |
| **F11** | **Oracle absent / inconclusive** | no deterministic check applies | gate detects `Inconclusive` (§4.6.4) | self-consistency vote + LLM-judge fallback (gated) | bounded vote K |
| **F12** | **Verifier gamed** | candidate passes a weak check but is wrong | rank oracles strictly; trajectory/PRM penalty; rubric verifier | deterministic oracles dominate; suspicious-path penalty (§4.8.4) | scoring policy |
| **F13** | **Runtime crash/hang mid-run** | runtime 5xx / stall | `max_stall_ms` watchdog + ch.01 `RuntimeSupervisor` | pause run; restart runtime; resume from checkpoint (§4.13) | watchdog + supervisor |
| **F14** | **Budget exhaustion** | run hits a cap | Governor (§4.3.2) | finalize honestly with partial result + lessons; never silent | hard caps |
| **F15** | **Subagent runaway / no-return** | child loops or never finishes | child Governor + parent `deadline` (§4.10.4) | child self-aborts to `partial`; parent never blocks | `max_subagents`, child budget |
| **F16** | **Replay divergence** | replayed state ≠ recorded | determinism CI: replay must reproduce events (K2) | quarantine non-determinism (temp/network are *recorded*) | seed-pinning |
| **F17** | **Plan cycle / unschedulable** | DAG has a cycle / deadlock | `dag.acyclic()` guard at `Plan` (§4.5.2) | reject plan; re-synthesize under schema constraint | `max_replans` |
| **F18** | **Cascading hallucination** | one wrong fact poisons downstream | provenance on every span (ch.04); per-step oracle isolates blast radius | step-local verification catches before propagation (K1) | per-step gate |
| **F19** | **Destructive action** (rm -rf, force-push) | irreversible effect proposed | capability gate + approval for destructive class (§4.3.3) | `Paused` for approval even in `auto-apply` | autonomy policy |
| **F20** | **Multi-agent conflict** (two children edit same file) | merge collision | worktree isolation (§4.10.2) + oracle-gated merge | conflicting child re-verified post-merge or rejected | isolation (K9) |

**The meta-point:** the most dangerous failures (F5 broken build, F10 premature done, F7 thrash, F1 loops) are made **structurally impossible** by the design — the gate, the idempotency cache, and the per-file/step/run budgets — rather than merely *handled*. This is what "the loop is finished" means: the failure modes are enumerated and each is either prevented by construction or has a bounded, recorded recovery.

---

## 7. Extensibility / plugin points

Every part of the kernel is an extension seam (ch.01 §7 manifest + capability negotiator). To add a capability, *no file under `hide-kernel/` core changes*.

| Seam | Trait / contract | What a plugin contributes | Capability gate |
|---|---|---|---|
| **Oracle** | `Oracle` (§4.6.1) | a new verifier (e.g. a security scanner, a perf-regression check, a custom test runner) | `verify:register` |
| **Strategy** | `Strategy` (§4.8) | a new search/sampling strategy (e.g. a domain-specific MCTS) | `search:register` |
| **Tool** | tool registry + `ToolCall`/`ToolResult` (§4.9) | a new action (e.g. `db.migrate`, `k8s.apply`) with arg schema | `tool:<name>` grant |
| **ContextSource** | ch.04 `ContextSource` | a new context provider feeding plans/steps | `context:source` |
| **MemoryStore / Skill backend** | ch.04 `MemoryStore` | an alternative skill/lesson store (e.g. team-shared) | `memory:store` |
| **Planner** | `planner.rs` synthesis hook | a custom plan-synthesis policy (e.g. PDDL-backed) | `plan:planner` |
| **Autonomy policy** | `autonomy.rs` | custom approval gates (e.g. enterprise "no force-push ever") | policy-layer (locked) |
| **Subagent kind** | `SubagentSpec.kind` + return contract | new delegation roles | `subagent:spawn` |
| **Model provider** | ch.01 `ModelProvider` | a different model behind the loop (the loop is provider-agnostic) | `provider:register` |
| **Event subscriber** | ch.01 event bus | observe the trajectory (telemetry, custom UI) | `events:subscribe` |

The litmus test (ch.01): *"to add capability X, does anyone edit `hide-kernel/`?"* — for all of the above, no. A new oracle is the canonical example: drop in a `cargo audit` oracle, register it under `verify:register`, reference it in a plan step's `acceptance.oracles`, and the gate runs it — zero core change.

---

## 8. Bleeding-edge / moonshots (ranked)

Ranked by (impact ÷ difficulty), each tagged maturity. These are the post-v1 frontier; v1 is the deterministic loop above.

1. **`fine-tune-at-Condense` protocol training [SPECULATIVE, high impact, med difficulty].** Co-train the served model (via *Hawking Condense*) to natively emit HIDE's plan/tool-call protocol and to be a strong self-drafter for agent scaffolds. Turns "prompt the model to follow our protocol" into "the model *is* our protocol." The deepest moat; gated on Condense maturity.
2. **A trained Process-Reward Model as a local oracle [RESEARCH-PROVEN, high impact, high difficulty].** A small PRM (Condense-quantized) scoring trajectories step-by-step would sharpen best-of-N/LATS selection beyond self-consistency — and it runs free, locally. The verifier-quality ceiling (§3.2) is the thing to raise; a good local PRM raises it.
3. **Entropy-forked search [SPECULATIVE, high impact, med difficulty].** Live next-token-entropy spikes auto-spawn ToT branches exactly at the model's decision points (§4.12.3). Spend search where doubt is, not uniformly — a uniquely-local capability.
4. **Self-improving harness (bounded Gödel-machine) [SPECULATIVE, high impact, very high difficulty].** The agent proposes improvements to its *own* skills/prompts/oracles, validated empirically before adoption (Darwin-Gödel-Machine, §3.4) — strictly sandboxed and oracle-gated. Powerful and dangerous; far out, behind a hard capability wall.
5. **Learned escalation controller [RESEARCH-PROVEN, med impact, med difficulty].** Replace the heuristic `pick_tier` (§4.8.5) with a policy learned from logged outcomes (which tier *actually* paid off for which step shape). Optimizes free-compute spend.
6. **Cross-session "team memory" [PROVEN-elsewhere, med impact, low difficulty].** Share the Skill Library across a team (a `MemoryStore` backend, §7) so a verified recipe one engineer's agent learned helps everyone's — opt-in, the user's data.
7. **Counterfactual replay / "what-if" debugging [SPECULATIVE, med impact, med difficulty].** Fork from any checkpoint with one upstream event changed and *deterministically* re-run downstream (§4.13) — a debugger for the agent's own decisions ("what if step 3 had used the other API?").
8. **Speculative *plan* execution [SPECULATIVE, med impact, high difficulty].** Begin executing the most-likely-needed next step (in a worktree) *before* the current step finishes verifying, discard if the prediction was wrong — spec-decode lifted from tokens to *plan steps*. Free compute makes the wasted work cheap.

---

## 9. Open questions / dials

The decisions deliberately left as **tunable dials** (with defaults), plus genuine open questions.

**Dials (config, ch.01 §4.10):**
- `max_steps`/`max_repairs`/`max_replans`/`max_wallclock_ms`/`max_subagents`/`max_edits_per_file`/`max_stack_depth` — the Governor caps (§4.3.1). Defaults: 80 / 3 / 4 / 30 min / 8 / 5 / 5.
- `search_breadth` (best-of-N N) / `search_depth` (ToT/LATS) / `self_consistency_k` — default 1/0/1 (ReAct); overnight profile bumps all.
- `EscalationPolicy` — when to climb the search ladder (§4.8.5).
- `Autonomy` — suggest-only ↔ autonomous (§4.3.3); destructive-action approval threshold.
- Oracle suite per `StepKind` — which deterministic oracles are mandatory (§4.6.2).
- `w_*` scoring weights for candidate selection (§4.8.4) and repair-vs-replan thresholds.
- `temperature`/`seed` strategy for best-of-N diversity (distinct seeds vs distinct temps).

**Open questions:**
- **Q1 — Optimal N for best-of-N as a function of model size / step difficulty?** More candidates help until the verifier can't distinguish them; the sweet spot is empirical and model-dependent. *Resolution: log outcome-vs-N and learn it (§8.5).*
- **Q2 — How aggressively to constrain plan decode?** Full grammar constraint guarantees validity but may *over*-constrain creativity; partial constraint risks invalidity. *Resolution: constrain *structure* hard, *content* soft; measure plan quality vs constraint tightness.*
- **Q3 — When does LATS pay off over best-of-N?** Tree search helps when *sequencing* matters (multi-step tool trajectories), not when a single artifact is the answer. The boundary is fuzzy. *Resolution: gate LATS to steps with `deps`-heavy sub-DAGs; measure.*
- **Q4 — LLM-judge trust budget.** How much weight may a probabilistic oracle carry as a tie-break before it starts overriding signal? *Resolution: cap its contribution below the deterministic-oracle margin (§4.8.4); A/B against oracle-only.*
- **Q5 — Skill-library staleness.** A repo evolves; a learned skill can rot. How fast to decay vs how aggressively to re-validate? *Resolution: re-validate a skill's worked example on retrieval; decay by fail_count (§4.11.2).*
- **Q6 — Determinism vs sampled search.** Best-of-N with distinct *temperatures* gives more diversity but breaks per-candidate replay unless seeds are pinned. We pin seeds (replay wins); the open question is whether temp-diversity's quality gain justifies a "diverse-but-replayable-as-recorded" mode. *Resolution: default seed-pinned; offer recorded-sample mode.*

---

## 10. Cross-references

- **ch.01 System Architecture** — the **Event envelope** (§4.6) our loop emits into (`turn.*`, `plan.*`, `token`, `tool.*`, `diff.*`, `verify.*`, `repair.*`, `subagent.*`, `approval.*`, `budget.transition`, `skill.*`); **Action/Observation classing** + the **replay-never-re-fires** rule (T3 = our K5); the **single-writer log / `seq` order**; the **`RuntimeSupervisor`** (F13); the **capability negotiator** (T4, every tool/oracle/subagent gate); the **extension manifest** (§7); the **on-disk layout** (`.hide/`, where checkpoints/skills live).
- **ch.04 Context & Memory** — the **Context Compiler** (we request per-step context), the **ContextManifest** (bound into checkpoints for byte-exact resume), the **ContextSource** trait (plan/step context providers), the **MemoryStore** (episodic lessons §4.7, procedural skills §4.11), the **retrieval scorer** (recency/importance/relevance, reused for skills/lessons), the **KvStore + KV checkpoint** (§4.13 warm resume), the **constrained-decode** substrate (`JsonConstraint`).
- **ch.03 Editor** (consumes) — renders our `plan.*` DAG (the plan-tree view), `diff.proposed`/`diff.applied` (the diff UX), `verify.result` (the test/build panel), the timeline scrubber (our replay, §4.13), and the approval prompts (`approval.requested`, §4.3.3).
- **ch.06 Runtime/Model** (provides/LATER) — the `logprobs`/entropy SSE field (§4.12.2–3), JSON-Schema/grammar constrained decode (§4.9.2), `POST /v1/hawking/kv/checkpoint` (§4.13), `fine-tune-at-Condense` protocol training (§8.1), spec-decode telemetry (§4.12.4).
- **`engine.rs` / `http.rs` / `json_constrain.rs` / `spec_gov.rs`** — the verified seams (see *Ground truth* in §1).

---

## Appendix A — Binding contracts

> **Normative.** Other chapters import these shapes. Additive-by-default (ch.01 T10): new fields/kinds may be added without a major bump; unknown fields round-trip.

### A.1 Plan & Step (the plan-as-data contract)

```jsonc
{
  "plan_id": "plan_<ULID>", "run_id": "run_<id>", "goal": "<string>",
  "status": "draft|active|done|failed|abandoned",
  "budget": { /* A.5 Budget */ },
  "steps": [{
    "step_id": "<string>", "parent": "<step_id|null>",
    "title": "<string>", "kind": "investigate|edit|command|verify|synthesize|decompose|delegate",
    "rationale": "<string>", "deps": ["<step_id>"],
    "status": "pending|active|done|failed|skipped|blocked",
    "acceptance": {                                  // THE VERIFIER CONTRACT — required
      "oracles": ["<oracle_id>"],                    // deterministic preferred
      "predicate": "<human-readable success condition>",
      "tests": ["<test selector>"],                  // optional, for `test` oracle
      "threshold": 0.7                               // optional, for probabilistic fallback
    },
    "produced": ["<artifact_ref>"],                  // outputs other steps consume
    "search_hint": { "tier": "react|best_of_n|tot|lats", "n": 4 },  // optional per-step override
    "attempts": 0, "repairs": 0, "est_tokens": 0, "actual_tokens": 0
  }]
}
```

### A.2 Oracle & Verdict (the verifier interface)

```rust
trait Oracle {
    fn id(&self) -> &str;
    fn class(&self) -> OracleClass;            // Deterministic | Probabilistic
    fn cost_hint(&self) -> Cost;
    fn check(&self, c: &Candidate, env: &VerifyEnv) -> Verdict;   // sandboxed; pure w.r.t. snapshot
}
struct Verdict {
    oracle: String,
    outcome: Outcome,                          // Pass | Fail | Inconclusive
    score: Option<f32>,                        // probabilistic only ∈ [0,1]; never overrides Deterministic
    failures: Vec<Failure>,                    // { file, line, code, category, message }
    artifacts: Vec<BytesRef>,                  // content-addressed (ch.01 blob store)
    duration_ms: u64,
}
// RULE: a Deterministic Verdict is AUTHORITATIVE; Probabilistic score only ranks within
//       the Deterministic-Pass set (the verifier-quality ceiling, §3.2 / §4.8.4).
```

### A.3 ToolCall & ToolResult (the tool-call protocol)

```jsonc
// tool.call (Action); tool.result (Observation, cause = call_id)
"ToolCall":   { "call_id","tool","args","capability_grant_id","idempotency_key",
                "expects":"diff_applied|text|exit_code|json","timeout_ms","dry_run" }
"ToolResult": { "call_id","ok","output","bytes_ref?","exit_code?","duration_ms",
                "side_effects":["<human description>"],"error?":{ "taxonomy_code","message" } }
// INVARIANTS: args validated against the tool's arg-schema (constrained-decode + lint, §4.9);
//   identical idempotency_key ⇒ deduped; in Replay mode dispatch returns the RECORDED result (K5).
```

### A.4 Subagent delegation contract

```jsonc
"SubagentSpec":   { "subagent_id","parent_run","goal","kind":"research|implement|verify|review",
                    "isolation":"none|context|worktree","budget":{…},"context_seed":[…],
                    "return_contract":{ "shape":"summary|diff|verdict|artifact_ref","schema":{…},
                                        "max_tokens":1500 },"deadline" }
"SubagentReturn": { "subagent_id","status":"ok|partial|failed|aborted","result":{…/*schema*/},
                    "summary":"<≤max_tokens — the ONLY thing entering parent context>",
                    "lessons":[…],"budget_used":{…},"artifacts":["<blob_ref>"] }
// INVARIANT: parent ingests only `summary`+`result` (clean-window discipline, §4.10).
```

### A.5 Budget (the governor contract)

```jsonc
"Budget": { "max_steps":80,"max_repairs":3,"max_replans":4,"max_wallclock_ms":1800000,
            "max_subagents":8,"max_stack_depth":5,"max_tool_calls":200,"max_edits_per_file":5,
            "token_budget_hint":0,"search_breadth":1,"search_depth":0,"self_consistency_k":1,
            "escalation":{ /* EscalationPolicy */ } }
// DEFINING CHOICE (K4): hard caps are wallclock/steps/effect-counts, NOT token spend.
```

### A.6 Skill (the persistent skill-library contract)

```jsonc
"Skill": { "skill_id","name","description","kind":"procedure|snippet|recipe|macro",
           "trigger","body":{ "steps":[…],"params":[…],"example_diff_ref" },
           "provenance":{ "learned_from_run","repo","commit" },
           "validation":{ "last_verified","oracle","success_count","fail_count" },
           "importance","last_access","access_count","supersedes","embedding_ref" }
// INVARIANT: only EXECUTION-VALIDATED solutions become skills (Voyager, §4.11); stored as the
//   user's file (.hide/memory/procedural/, ch.04); retrieved by recency/importance/relevance.
```

### A.7 Event kinds this chapter adds to the ch.01 registry

`plan.created` · `plan.step` · `plan.step_updated` · `plan.replanned` · `verify.result` · `repair.attempt` · `repair.lesson` · `search.expanded` · `search.selected` · `subagent.spawned` · `subagent.returned` · `approval.requested` · `approval.granted` · `approval.denied` · `budget.transition` · `skill.learned` · `skill.invoked` · `run.aborted`. (Each carries a registered JSON Schema per ch.01 §7.2; payloads as specified inline above.)

---

## Appendix B — Source register

**Core loops & search.** ReAct ([Yao et al. 2023], arXiv:2210.03629). Reflexion ([Shinn et al. 2023], arXiv:2303.11366). Tree-of-Thoughts ([Yao et al. 2023b], arXiv:2305.10601). Self-Consistency ([Wang et al. 2023b], arXiv:2203.11171). LATS ([Zhou et al. 2024], arXiv:2310.04406, ICML 2024). RAP ([Hao et al. 2023], arXiv:2305.14992). Plan-and-Act ([Erdogan et al. 2025], arXiv:2503.09572). Task-Decoupled / HiPlan long-horizon planning (arXiv:2601.07577, arXiv:2508.19076). RethinkMCTS for code (arXiv:2409.09584).

**Verification & reward.** Verifier-selected best-of-N ([Cobbe et al. 2021], GSM8K, arXiv:2110.14168). PRM survey ([Zhang et al. 2025], arXiv:2510.08049). Process Reward Models That Think ([Zhao et al. 2025], arXiv:2504.16828). Tool-integrated self-verification for small models / T1 ([Kang et al. 2025], arXiv:2504.04718). LLM-as-Judge ([Zheng et al. 2023b], arXiv:2306.05685) + bias-amplification caveat (arXiv:2505.19477). Agentic rubrics as verifiers (arXiv:2601.04171).

**Coding harnesses.** SWE-agent / ACI ([Yang et al. 2024], arXiv:2405.15793). OpenHands SDK ([Wang et al. 2025], arXiv:2511.03690; 72% SWE-Bench Verified). Aider (aider.chat docs). Cline/Roo (open-source VS Code agents). Context-as-a-Tool for long-horizon SWE (arXiv:2512.22087). Subtask-level memory for SWE agents (arXiv:2602.21611).

**Skills & procedural memory.** Voyager ([Wang et al. 2023c], arXiv:2305.16291). Memp / hierarchical procedural memory ([2025], arXiv:2508.06433, arXiv:2512.18950). Externalization-in-LLM-agents review (arXiv:2604.08224). SoK: Agentic Skills (arXiv:2602.20867).

**Multi-agent & failure.** AutoGen ([Wu et al. 2023], arXiv:2308.08155). Multi-agent debate ([Du et al. 2023], arXiv:2305.14325). Multi-LLM-Agents-Debate limits (ICLR-2025 blogpost; Tran & Kiela 2025). MAST failure taxonomy ([Cemri et al. 2025], NeurIPS 2025). TRAIL trace-reasoning (arXiv:2505.08638). LLM-agent hallucination taxonomy survey (arXiv:2509.18970). Where LLM agents fail & learn from failures (arXiv:2509.25370).

**Constrained decode.** XGrammar-2 (arXiv:2601.04426); grammar-constrained-generation reliability (constrained-decoding refs). In-tree substrate: `hawking-core/src/json_constrain.rs`.

**Generative Agents (retrieval scoring shape).** ([Park et al. 2023], arXiv:2304.03442).

**In-tree ground truth.** `hawking-core/src/engine.rs` (Engine trait, GenerateRequest{abort,json_mode,max_stall_ms}, SamplingParams{seed}, StreamEvent, GenStats{dec_tps,draft_*}, SpeculateMode{ExactShared,Eagle5}). `hawking-serve/src/http.rs` (routes incl. `/v1/hawking/generate`,`/v1/hawking/tokens`). `hawking-serve/src/spec_gov.rs`. `hawking-core/src/stateful/` + `system_kv_bank.rs` (consumed via ch.04).
