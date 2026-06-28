# 09 · Parallel Agents & Workstation Mode

> **Purpose (one line).** Turn the user's workstation into a **24/7, zero-marginal-cost agent cluster** — a scheduler, resource governor, isolation/merge layer, and remote-control protocol that runs *dozens* of ch.02 agent runs in parallel, overnight, and from a laptop driving a Mac Studio — doing the one thing cloud coding agents structurally cannot: spend unlimited local compute without a per-agent-hour bill.

**Status:** DESIGN. This chapter specifies the **orchestration/scheduling/merge/remote layer beneath many ch.02 runs**. It does **not** re-design the agent loop (ch.02 owns it), the event backbone (ch.01 owns it), or the sandbox/capability model (ch.10 owns it) — it *composes* them. Everything here is **SCOPED POST-SHELL**: the app shell (ch.01–03) ships first; parallel-agent and workstation features are a **later tier**, designed fully now so no further orchestration design is needed when the tier is built. The model/runtime is the stable localhost HTTP surface of ch.01 §4.3; `.tq`/32B serving is runtime testing, not shell-gating.

**Tier tags used throughout:** **[SHELL-FIRST]** = belongs to the base app (rare in this chapter); **[TIER-2: SWARM]** = the parallel-agent layer (the bulk); **[TIER-3: WORKSTATION]** = the laptop→Mac-Studio remote-server mode; **[TIER-4: CLUSTER]** = optional cross-machine distribution. Within each, **[PROVEN]/[RESEARCH-PROVEN]/[SPECULATIVE]** carry *difficulty* (build cost for us) and *impact*.

---

## Table of contents

1. [Purpose & scope](#1-purpose--scope)
2. [Tenets](#2-tenets)
3. [State of the art + limits (cited)](#3-state-of-the-art--limits-cited)
4. [The Hawking design (concrete)](#4-the-hawking-design-concrete)
   - 4.1 [Module layout & where this sits](#41-module-layout--where-this-sits)
   - 4.2 [Orchestration patterns & when each wins](#42-orchestration-patterns--when-each-wins)
   - 4.3 [The isolation model (worktrees, FS/process/network)](#43-the-isolation-model)
   - 4.4 [Merge & conflict resolution (N attempt, judge selects/merges)](#44-merge--conflict-resolution)
   - 4.5 [The task queue & job schema](#45-the-task-queue--job-schema)
   - 4.6 [The scheduler & the resource Governor](#46-the-scheduler--the-resource-governor)
   - 4.7 [Overnight / batch jobs (checkpointed, resumable, report-on-wake)](#47-overnight--batch-jobs)
   - 4.8 [Parallel testing & parallel research](#48-parallel-testing--parallel-research)
   - 4.9 [Workstation / remote mode (the wire protocol)](#49-workstation--remote-mode)
   - 4.10 [Cross-machine distribution (optional)](#410-cross-machine-distribution-optional)
5. [How we EXCEED](#5-how-we-exceed-cloud-literally-cannot-do-this)
6. [Failure modes + mitigations](#6-failure-modes--mitigations)
7. [Extensibility (new orchestration patterns)](#7-extensibility-new-orchestration-patterns)
8. [Bleeding-edge / moonshots (ranked)](#8-bleeding-edge--moonshots-ranked)
9. [Open questions / dials](#9-open-questions--dials)
10. [Cross-references](#10-cross-references)
- [Appendix A — Binding contracts (schemas other chapters import)](#appendix-a--binding-contracts)
- [Appendix B — Source register](#appendix-b--source-register)

---

## 1. Purpose & scope

This chapter answers one design question:

> *Given many independent ch.02 agent runs that each cost nothing to run, how do we schedule, isolate, merge, govern, and remotely drive them so the workstation behaves like a private, always-on team of engineers — without ever thrashing the machine, deadlocking, or merging garbage?*

The thesis: **the agent loop is the unit; this chapter is the fabric that runs hundreds of them.** ch.02 made *one* run reliable (plan-as-data + oracle-first verification + bounded search + best-of-N over isolated worktrees with an oracle selector). The unfair advantage of *local* is that one such run costs $0 and hits no rate limit — so the natural move is to run **dozens at once and overnight**. But "dozens of agents" on one box is a *systems* problem: a queue, a DAG scheduler, a resource governor that respects Apple-Silicon RAM/thermal reality, an isolation model so parallel edits never collide, a merge layer so N attempts become one verified result, and a remote protocol so a laptop can drive a Mac Studio server. That fabric is this chapter.

### In scope

- **Orchestration patterns** — fan-out/map-reduce, pipeline, tournament/best-of-N-with-judge, debate, planner→workers→merger — with the *decision rule* for when each wins (and when single-agent wins, per ch.02 K12).
- **The isolation model** — git worktree per agent; filesystem/process/network/port isolation; how parallel edits avoid collision; coordinated with ch.10's sandbox.
- **Merge & conflict resolution** — detection, three-way/structured merge, the integration-branch funnel, and the "N agents attempt → judge selects-or-merges" flow built on ch.02's oracle gate.
- **The task queue & scheduler** — priorities, dependency DAG, fair-share, preemption/eviction, backpressure, with a normative job/queue schema.
- **The resource Governor** — GPU/RAM/thermal admission control on Apple-Silicon unified memory; extends ch.02's per-run Governor to a *machine-wide* governor.
- **Overnight/batch jobs** — long-running, checkpointed, resumable across reboots, report-on-wake.
- **Parallel testing & parallel research** — sharded test execution and orchestrator-worker research fan-out.
- **Workstation/remote mode** — laptop thin-client → Mac-Studio agent server: the wire protocol, auth, state sync, streaming, reconnection, security posture (coordinated with ch.10).
- **Cross-machine distribution** — optional multi-box scheduling.

### Out of scope / delegated (with the binding it uses)

| Item | Owner | This chapter's relationship |
|---|---|---|
| The agent loop, the per-run Governor, the SubagentSpec/Return contract, best-of-N + oracle selection | **ch.02** | We **extend** the per-run Governor to a machine-wide one and **schedule** many runs; we never redefine the loop or the subagent contract. |
| The Event envelope (`seq`/`run_id`/`parent`/`cause`, Action/Observation, replay-never-re-fires) | **ch.01 §4.6** | Every parallel run emits these unchanged; our scheduler/merge emit **new kinds** (`job.*`, `merge.*`, `remote.*`) registered per ch.01 §7.2. |
| The sandbox + capability model (what an agent may touch) | **ch.10** | Our isolation **references** ch.10's sandbox; we add the *orchestration* on top (worktree lifecycle, port leasing). We do not invent a second security model. |
| Context packing / retrieval / memory | **ch.04** | Subagents get minimal context seeds via ch.04; parallel research summaries fold into ch.04 memory. |
| Sampler/grammar/KV kernels, `.tq`/32B serving | **ch.06 + Hawking Condense** | The scheduler consumes the runtime's `max_batch_size` and `/metrics`; it never reaches inside the engine. |

> **Scoping invariant (restated).** Nothing in TIER-2/3/4 blocks the shell. A single interactive ch.02 run is the SHELL-FIRST product. This chapter's machinery activates only when the user runs **more than one** run at once, or starts a **batch**, or connects a **remote client** — each an explicit, opt-in escalation.

### Ground truth this chapter binds to (verified in-tree)

- **`crates/hawking-serve/src/lib.rs`** — `RuntimeOpts.max_batch_size` (default 1) drives a **continuous-batching loop** with a per-slot scheduler and a `should_gather(ready, max_batch)` admission predicate; `ServeOpts` logs the per-step cost (`B×vocab×4` full-logits vs `B×4` greedy-lane bytes). **This is the runtime's own concurrency ceiling** — the machine-wide scheduler (§4.6) must never admit more *concurrent generations* than the runtime can batch without RAM blow-up, and reads this number as a hard input.
- **`hawking-serve` HTTP surface** (ch.01 §4.3) — N concurrent SSE streams over `/v1/hawking/generate|tokens`, `/healthz`, `/metrics`. Each session worker holds its own stream (no pipe contention) — this is *why* many parallel runs are even possible on one runtime.
- **`GenStats::dec_tps()`, `spec_gov` accept counters** via `/metrics` — the Governor's telemetry inputs (degradation detection, escalation bias).
- **`GenerateRequest.abort: Option<Arc<AtomicBool>>`** — the cooperative-cancel handle the scheduler flips on preemption/eviction (ch.02 §4.4).
- **`SamplingParams.seed`** — determinism handle; parallel runs that must be reproducible pin it (T6).

Apple-Silicon reality this chapter *designs around* (cited §3): **unified memory is the binding constraint** (Metal caps GPU residency at ~75% of RAM; swap = throughput collapse), and **sustained inference thermally throttles 20–40% after 10–15 min**. The Governor treats RAM-headroom and a thermal signal as first-class admission inputs — not afterthoughts.

---

## 2. Tenets

Twelve tenets. Each extends ch.01 (T1–T10) and ch.02 (K1–K12) into the *multi-run* domain. Every later decision cites one.

| # | Tenet | Consequence |
|---|-------|-------------|
| **P1** | **The machine is the budget; tokens are free.** The limiting reagents are **RAM, GPU time, and thermal headroom** — never dollars. The scheduler optimizes throughput under a *physical* resource envelope. | The Governor admits on RAM/GPU/thermal, not token count (extends ch.02 K4 from per-run to machine-wide, §4.6). |
| **P2** | **One run is the atom; the fabric never reaches inside it.** Every scheduled unit is a complete ch.02 run with its own Governor, plan, and event stream. | We compose runs; we never fork the loop's internals. The scheduler sees a run as a black box with a budget and a status (§4.1). |
| **P3** | **Isolation by default; merge through verification only.** Parallel work runs in isolated worktrees; results re-enter shared state **only** after ch.02's oracle gate passes and the merge is conflict-clean. | Worktree-per-run; no parallel run ever writes the user's working tree directly (§4.3, §4.4; ch.02 K1/K9). |
| **P4** | **The verifier is the selector.** When N runs attempt the same goal, **deterministic oracles rank first**; a judge only breaks ties among oracle-passing candidates (ch.02 §4.6 ceiling, applied to selection). | Tournament/best-of-N selection is oracle-first; the LLM-judge never overrides a passing build/test (§4.4). |
| **P5** | **Single-agent first; parallelism earns its place.** Coordination adds failure modes (deadlock, drift, conformity); we fan out only for **genuinely separable, independently-verifiable** work, or for **best-of-N on one hard goal**. | The pattern-selection rule (§4.2) gates every topology; the default for an ambiguous task is *one run* (ch.02 K12; the 2025 multi-agent reckoning, §3). |
| **P6** | **Bounded swarms, structurally.** Every queue, fan-out width, worktree count, port lease, and concurrent-generation count has a hard ceiling enforced at one chokepoint. | No runaway swarm. The machine-wide Governor is the single admission gate (§4.6; ch.02 K8). |
| **P7** | **Backpressure everywhere; no unbounded queue.** Every producer→consumer hop is a bounded channel with a defined overflow policy; admission *blocks* rather than over-commits the machine. | Inherited from ch.01 T8; the queue admits only what the Governor's envelope allows (§4.5/§4.6). |
| **P8** | **Durable, replay-safe, resumable across reboots.** A batch started at midnight survives a crash/reboot and reports on wake; replay folds recorded effects, never re-fires them. | Jobs are events; the queue is a projection of the log; resume = replay-to-tail + relaunch-incomplete (ch.01 T1–T3; §4.7). |
| **P9** | **Fair-share by default; priority is explicit.** Interactive runs preempt batch; no single run starves the queue; the user's *foreground* turn always wins. | Weighted fair-share + priority classes + preemption (§4.6). |
| **P10** | **The remote plane is the same plane.** A laptop driving a Mac-Studio server speaks the *same* intent-in/events-out model as the local WebView; the server is authoritative; the client is a disposable view. | One protocol, local or remote; reconnection resumes from `seq` (ch.01 T2; §4.9). |
| **P11** | **Remote is closed by default.** The agent server binds loopback; remote access is opt-in, authenticated, encrypted (wss/TLS or SSH tunnel), and capability-scoped per ch.10. | Security posture is deny-first; §4.9 references ch.10's capability model, never replaces it. |
| **P12** | **Observability is mandatory for swarms.** A 30-agent overnight run is unmanageable without a live fleet view, per-run cost/resource accounting, and a wake report. | Every job emits resource + outcome telemetry; the fleet view is a projection (§4.7, §5; ch.01 T9). |

---

## 3. State of the art + limits (cited)

Tagged **[PROVEN]/[RESEARCH-PROVEN]/[SPECULATIVE]** with *difficulty*/*impact*. Full register in [Appendix B](#appendix-b--source-register).

### 3.1 Orchestration frameworks & patterns

| System / pattern | Mechanism (compressed) | Lesson for HIDE | Maturity |
|---|---|---|---|
| **LangGraph** | Graph of nodes; **checkpointed** state at every superstep; durable execution = resume from last checkpoint; Postgres/DynamoDB savers in prod. | Largest production footprint 2026; **checkpoint-at-superstep + resume** is the durability model — we adopt it, but our checkpoint *is* the event log (ch.01), not a side DB. ([LangGraph durable execution]) | [PROVEN] |
| **CrewAI** | Role-based agents (planner/researcher/writer) with shared memory; great demo ergonomics. | "Roles" map to our planner→workers→merger pattern; CrewAI's weakness is *production observability/error-recovery* — which our event log + Governor supply. ([Multi-agent frameworks 2026]) | [PROVEN] |
| **OpenAI Swarm → Agents SDK** | **Handoffs** (explicit control transfer carrying context) + guardrails + tracing; Swarm was educational, Agents SDK (Mar 2025) is the prod successor. | Handoff = our **pipeline** pattern (§4.2); the SDK's guardrails+tracing validate that *bounds + observability* are non-optional. **Mar 2026: Agents SDK + Temporal integration** for durable execution — confirms the durability direction. ([OpenAI Agents SDK orchestration]) | [PROVEN] |
| **AutoGen / debate** | Multiple agents converse/critique to consensus. | Our **debate** pattern (§4.2) — but gated hard by the 2025 reckoning below. ([AutoGen]) | [PROVEN, caveated] |
| **Pattern language (producer/consumer/critic/judge + coordinator)** | A *coordinator* routes work between producers, consumers, critics, judges; handles fan-out/fan-in, retry, termination — "the plumbing that makes archetypes composable." | This is **exactly our scheduler's role**: it routes, sequences, retries, terminates — it does **not** generate or judge. Framework-agnostic, maps onto Claude-subagents/LangGraph/CrewAI/Agents-SDK alike. ([Multi-agent orchestration pattern language 2026]) | [PROVEN] |

> **The five production patterns** that recur across every source (LangGraph, CrewAI, Agents SDK, the pattern-language writeups): **fan-out, pipeline, debate, supervisor, swarm** — each with a sharp best-fit *and* a sharp anti-pattern. §4.2 adopts this set and adds **tournament/best-of-N** (from the SWE-agent literature) and **map-reduce** (fan-out + a reduce step) as first-class.

### 3.2 Parallel coding agents & best-of-N selection (the direct evidence)

- **Parallel rollouts + best-of-N with a selector is the dominant test-time-scaling shape for SWE agents.** Run *k* independent agents on the same task; select the best patch. Production recipe (2025–2026 SWE-bench work): **sample 5–10 trajectories per problem**, run **regression tests on every candidate, filter out regressors, then majority-vote / judge** the survivors. ([SWE-Master post-training], [SWE-Replay test-time scaling], [DeepSWE]) → This is **literally §4.4's selection pipeline** — and HIDE gets it *for free* (no per-trajectory API bill). **[RESEARCH-PROVEN]**, difficulty *low* (we already have the oracle gate), impact *high*.
- **SWE-agent batch mode** runs many instances over a shared config pool with per-instance isolation. ([SWE-agent batch mode]) → validates a **shared config + isolated execution** scheduler shape (§4.5/§4.6).
- **Massive parallel infra at training time**: RL iterations spawn **512 Docker containers in parallel** with Kubernetes scheduling across a node pool. ([DeepSWE], [SWE-Master]) → the *upper bound* of the cluster idea (§4.10); HIDE's single-box version is dozens, not hundreds, bounded by RAM.

### 3.3 Anthropic's orchestrator-worker (the architecture we mirror)

- **Claude Research = orchestrator-worker**: a lead agent plans, spins up **3–5 specialized subagents in parallel each with its own context window**, then a separate citation/synthesis pass. Beat single-agent Opus 4 by **90.2%** on the internal research eval — but at **~15× the tokens**. **Token usage explains 80% of the performance variance.** ([Anthropic multi-agent research system]) → Two load-bearing lessons: (1) **parallel subagents with isolated context windows are the right shape for breadth-first research** (§4.8) — and the 15× token cost that gates it for cloud users is **free for us**; (2) **subagents need crisp objectives/boundaries/output-format or they duplicate work or leave gaps** — exactly what ch.02's `SubagentSpec.return_contract` enforces. **[PROVEN]**, difficulty *low* (ch.02 contract exists), impact *high*.

### 3.4 Git-worktree-based parallel agents (the isolation primitive)

- **Worktrees are the load-bearing isolation primitive for parallel coding agents in 2026.** Each agent gets its own working directory off the same `.git`, so parallel sessions never overwrite each other. **Conductor** (Mac app, Melty Labs) creates a worktree per workspace + diff/PR flow; **Crystal/Nimbalyst** runs parallel Claude/Codex sessions with visual management; **ccswarm** coordinates specialized pools in worktree-isolated envs; VS Code added worktree support (Jul 2025). ([Conductor], [Crystal/Nimbalyst], [Augment worktree guide], [Upsun worktrees]) → §4.3 adopts worktree-per-run as the default.
- **Code isolation ≠ runtime isolation.** Worktrees isolate *files* but not *ports/processes/databases*: "Port conflicts hit first — every dev server defaults to 3000/5432/8080; launch two and one fails." **Dagger container-use** combines worktrees with container isolation to close the gap. ([Penligent runtime isolation], [container-use]) → **§4.3 must lease ports + namespace processes/env**, not just fork the tree; this is a *named* gap we design for.
- **Merge is the hard part.** Even with isolated trees you resolve conflicts manually at merge; the recurring working pattern is **a staging/integration branch: merge all feature branches there, run tests, fix conflicts, then merge the clean result to main**; plus a **shared task doc** so agents claim disjoint work. Git's recursive 3-way merge handles two heads; **octopus refuses on any conflict**. ([git-merge docs], [merge-strategies], [Nick Mitchinson worktrees], [MindStudio worktrees]) → §4.4 builds exactly this funnel and adds **footprint-disjointness scheduling** to *prevent* conflicts up front.

### 3.5 Task queues, DAG schedulers & backpressure

- **Standard distributed-scheduler decomposition**: a **Scheduler** (decides what runs when), a **Queue** (buffers work), **Workers** (execute). DAG dependencies via **Kahn-style topological progression** — a node becomes ready when its in-degree hits 0. Priority via **min-heap**; **weighted fair-share** for fairness; **Kubernetes-style priority classes + preemption** (evict lower-priority pods when a high-priority one needs to run). ([System Design: distributed job scheduler], [algomaster], [GeeksforGeeks task queue]) → §4.5/§4.6 adopt this skeleton verbatim, sized for one box.
- **Backpressure in a parallel DAG executor (Rust)**: keep the **results channel unbounded so completed tasks always report back** (forward progress), but **bound the admission side** so you never over-commit. ([reymom backpressure in parallel executor]) → §4.6's exact policy: *bounded admission, unbounded completion-reporting*.
- **Idempotency is a prerequisite for safe checkpointing**: any tool that writes external state needs an idempotency key tied to workflow state so replay doesn't double-fire. ([Temporal durable execution], [Diagrid checkpoints-vs-durable]) → ch.01 T3 already gives us this (effects recorded as outcomes); §4.7 leans on it for resumable batches.

### 3.6 Durable execution & long-running agents

- **Checkpoints alone are not durable execution.** LangGraph checkpoints state at supersteps; **Temporal** replays event history to reconstruct state and resume *at the exact step* after a crash, with **Continue-As-New** to bound history growth. ([Temporal vs LangGraph], [Diagrid]) → HIDE is **closer to Temporal**: the event log *is* the history; resume = replay-to-tail. Our "Continue-As-New" analogue is ch.01's snapshot+segment-archival (§4.7).
- **Durable sessions outlive any one connection.** AI-streaming best practice (2025–2026): **server-side sessions persist independently of the socket; reconnect at the last-acked offset with no duplicate tokens** (SSE `Last-Event-ID`-style resumption); "a 5-minute task that drops at minute 4 resumes at minute 4." ([WebSocket.org AI streaming], [Agent Client Protocol streamable-HTTP/WS]) → §4.9's reconnection model is exactly this, keyed on `seq`.

### 3.7 Remote agent control & the Agent Client Protocol

- **Agent Client Protocol (ACP)** — "the LSP for coding agents": an open **JSON-RPC 2.0** standard connecting any editor to any agent, **for both local and remote**. Zed shipped it Aug 2025; JetBrains partnered Oct 2025; community clients for Neovim/Emacs/VS Code. **Session model**: `session/new` (can declare MCP servers in the same handshake), streaming updates, `session/load` to resume. ([Agent Client Protocol], [Zed ACP], [Kiro ACP]) → HIDE's remote protocol (§4.9) is **ACP-shaped** (JSON-RPC, session-centric, resumable) so a HIDE laptop client could one day drive *any* ACP agent and vice-versa — but our wire carries the **ch.01 Event envelope** as the update payload, which is richer (causal DAG, replay-safe) than ACP's session-update notifications.
- **Codex/OpenClaw remote architecture (2026)**: an **app-server** the TUI connects to over **SSH-tunneled WebSocket**; "the server keeps working when the TUI disconnects; the TUI resynchronises on reconnect; resumed sessions preserve transcript, plan history, and prior approval decisions." Auth tokens only over `wss://` or loopback `ws://`; loopback + SSH port-forward preferred for plain WS. ([Codex remote connections], [OpenClaw remote]) → §4.9 adopts **server-authoritative + reconnect-resync + SSH/wss + loopback-default** wholesale.

### 3.8 The sobering limits (the policy these set)

- **Multi-agent debate does not reliably beat single-agent test-time compute at equal budget.** When compute is normalized, single agents **match or exceed** multi-agent on multi-hop reasoning (Qwen3, DeepSeek-R1-Distill, Gemini 2.5); an information-theoretic argument (Data Processing Inequality) says a single agent with perfect context utilization is *more information-efficient*. Failure modes named: **over-exploration/drift** (agents wander into sub-questions and lose the goal) and **majority-pressure conformity** (agents converge to consensus instead of deliberating). ([Single-agent vs multi-agent equal-budget 2026], [Multi-LLM-Agents-Debate ICLR-blog 2025], [Stop Overvaluing Multi-Agent Debate 2025]) → **HIDE policy: debate is a *fallback* pattern, not a default; best-of-N with an *oracle* selector beats debate because the selector is ground truth, not a vote** (P4/P5).
- **MAST taxonomy** (NeurIPS 2025, 1,600+ traces): 14 multi-agent failure modes — **specification/design 41.8%, inter-agent misalignment 36.9%, verification gaps 21.3%.** ([Cemri et al. 2025]) → §6 maps each bucket to a mitigation; the punchline is that *most* multi-agent failure is **bad task spec + missing verification**, both of which ch.02's plan-as-data + oracle gate already attack — so our parallel layer inherits the cure.
- **Apple-Silicon physical ceiling**: Metal caps GPU residency ~75% of unified RAM; **two concurrent generations share compute** (don't add tok/s); **swap collapses throughput**; **thermal throttling −20–40% after 10–15 min** sustained. ([Apple-silicon LLM limitations], [MACGPU concurrency/queue 2026], [SolidAITech unified memory]) → §4.6's Governor is built *on* these numbers: concurrency is bounded by **RAM headroom and the runtime's `max_batch_size`**, not by core count, and a **thermal backoff** is a first-class scheduler input.
- **Runaway-agent cost circuit-breakers** (cloud framing: "a runaway loop is a massive bill") — loop detection, EWMA anomaly detection, max-N-attempts-per-item, min-gap-between-attempts, wall-clock timeboxing, ceilings-per-resource. ([Cost circuit breaker], [Oracle runtime budget guardrails]) → For us the bill is *thermal/RAM*, but the *mechanisms* port directly into §4.6/§6 (the machine-wide Governor *is* the circuit breaker).

> **Synthesis that sets this chapter's policy.** The literature converges: (1) **parallel best-of-N with a verifier scales reliability** and is the dominant SWE-agent shape — and we have it *for free*; (2) **orchestrator-worker with isolated context windows is the right breadth pattern** (Anthropic's 90.2%) — and its 15× token cost is *free* for us; (3) **debate/consensus is overrated** — prefer oracle selection; (4) **worktrees isolate files but not runtime** — we must lease ports/namespace processes; (5) **merge is the hard part** — use an integration-branch funnel + footprint-disjoint scheduling; (6) **the machine, not money, is the budget** — the Governor admits on RAM/GPU/thermal. §4 is the build of exactly this.

---

## 4. The Hawking design (concrete)

### 4.1 Module layout & where this sits

A headless Rust crate `hide-fleet`, hosted in-process by the Tauri host (ch.01 §4.1), *above* `hide-kernel` (ch.02). It treats a ch.02 run as a black box: it **enqueues** runs, **schedules** them under the Governor, **isolates** them (worktree + sandbox lease), **merges** their verified results, and **serves** them to a remote client. It links no GPU code and never reaches inside the loop (P2).

```
hide-fleet/                              # the orchestration fabric ABOVE ch.02 (headless)
  src/
    lib.rs                               # FleetManager: owns the queue, scheduler, governor, registry
    queue/                               # THE TASK QUEUE (§4.5)
      schema.rs                          # Job, JobSpec, JobStatus, JobGraph (Appendix A.1)
      store.rs                           # durable queue = projection of the event log (ch.01)
      dag.rs                             # Kahn topological ready-set; cycle detection
      admission.rs                       # backpressure: admit iff Governor envelope allows (P7)
    sched/                               # THE SCHEDULER (§4.6)
      scheduler.rs                       # priority + fair-share + dependency dispatch loop
      governor.rs                        # MACHINE-WIDE resource governor (RAM/GPU/thermal) (§4.6)
      preempt.rs                         # preemption/eviction (checkpoint-and-yield)
      resources.rs                       # ResourceProbe: RAM headroom, thermal, dec_tps, max_batch
    isolate/                             # ISOLATION MODEL (§4.3)  [REFERENCES ch.10 sandbox]
      worktree.rs                        # git worktree lifecycle (create/prune/gc); overlay-FS option
      ports.rs                           # PortLease allocator (avoid 3000/5432/8080 collisions)
      env.rs                             # per-run env/namespace seed (TMPDIR, caches, DB names)
    merge/                               # MERGE & CONFLICT RESOLUTION (§4.4)
      plan.rs                            # footprint analysis → disjoint vs overlapping classification
      integration.rs                     # the staging-branch funnel (merge→test→promote)
      resolve.rs                         # 3-way + structured (tree-sitter) merge; conflict events
      select.rs                          # tournament selector: oracle-first, judge tie-break (P4)
    patterns/                            # ORCHESTRATION PATTERNS (§4.2)
      pattern.rs                         # Pattern trait + the selection rule (when each wins)
      fanout.rs map_reduce.rs pipeline.rs tournament.rs debate.rs planner_workers.rs
    batch/                               # OVERNIGHT/BATCH (§4.7)
      batch.rs                           # BatchJob: checkpointed, resumable, report-on-wake
      report.rs                          # the wake report (what ran, what passed, what to review)
      schedule.rs                        # cron-like "run at 02:00" triggers (idle/plugged-in gates)
    remote/                              # WORKSTATION/REMOTE MODE (§4.9)  [REFERENCES ch.10 auth]
      server.rs                          # the agent-server endpoint (wss/JSON-RPC, ACP-shaped)
      session.rs                         # server-authoritative session; reconnect-resync on seq
      protocol.rs                        # the wire envelope (Appendix A.4) — carries ch.01 Events
      auth.rs                            # token/mTLS handshake; loopback-default; SSH-tunnel mode
      pair.rs                            # device pairing (QR/code), trust store
    cluster/                             # CROSS-MACHINE (optional, §4.10) [TIER-4]
      pool.rs                            # node pool, capability advertise, cross-box dispatch
    fleetview.rs                         # the live fleet projection (per-run resource + status) (§5)
```

> **Why a crate above the kernel, not inside it.** The kernel must stay testable as *one* run; the fabric must be testable as *many mocked* runs without a model. `hide-fleet` mocks `hide-kernel` with scripted run-outcomes and asserts the exact `(schedule, governor-decision, merge-result)` sequence — the same property-test discipline ch.02 uses for the loop. The fabric is also reusable headless by a CLI (`hide swarm`, `hide batch`).

The four-tier process picture (ch.01 §4.2) is unchanged; `hide-fleet` lives inside the Tauri host alongside `hide-kernel`. The only *new* process surfaces are: (a) the **worktree-isolated runs** (still session-worker tasks, just rooted in different directories) and (b) the **remote server endpoint** (a bound socket, §4.9). The runtime sidecar is shared by all runs via continuous batching.

---

### 4.2 Orchestration patterns & when each wins

Seven patterns, each a `Pattern` implementation that composes ch.02 runs. **The selection rule (P5) gates them all**; the default for an ambiguous task is *one* run (ch.02 K12). A pattern is chosen by the **planner** (ch.02 §4.5, as a plan-level `orchestration` hint) or by the user (a profile/intent), never auto-escalated past the Governor's ceiling (P6).

#### 4.2.1 The catalogue

| Pattern | Shape | Wins when | Anti-pattern (don't) | Selector / merge |
|---|---|---|---|---|
| **Single-agent** *(default)* | one ch.02 run | task is sequential, exploratory, or ill-specified; **anything where you can't write the acceptance oracle up front** | forcing parallelism on a task that needs one coherent context (the equal-budget single-agent win, §3.8) | n/a |
| **Fan-out / map-reduce** | planner splits into **footprint-disjoint** subtasks → N worktree runs → **reduce** step merges | the work *partitions cleanly* (refactor 8 files, port 12 endpoints, migrate N modules); each part independently verifiable | parts share state / edit the same lines → merge hell; use single-agent or sequence instead | reduce step = ch.02 run that integrates + runs the *full* suite (§4.4) |
| **Pipeline** | run A → run B → run C; each consumes the prior's verified output (ch.02 `Delegate` + handoff) | staged work with clean handoffs (design → implement → test → doc); each stage has a distinct oracle | stages that need to iterate *together* (tight design↔impl loop) → keep in one run | each stage's oracle gates the handoff; no merge (sequential on one tree or chained worktrees) |
| **Tournament / best-of-N** | N runs attempt the **same** goal in **isolated** trees → **oracle-first selection** → adopt the winner | one **hard, well-specified** goal where the first attempt often fails (tricky bug, perf-sensitive patch); you *can* write the oracle | no usable oracle (then it's a vote = noise); or trivial goal (waste) | **the §4.4 selection pipeline** (regression-filter → oracle-rank → judge tie-break) (P4) |
| **Planner → workers → merger** | a lead run plans + spawns specialized workers (research/impl/verify) → a merger run synthesizes | breadth tasks needing *isolated context windows* (Anthropic's 90.2% shape, §3.3); large investigations | over-decomposition (MAST's 41.8% spec failures) → workers duplicate/gap; needs crisp `return_contract` | merger = ch.02 run folding `SubagentReturn` summaries (ch.02 §4.10) |
| **Debate / critic panel** *(fallback)* | K runs propose; a critic round reconciles | genuinely subjective/under-specified synthesis where *no* oracle exists **and** diversity matters | **default reach** — the 2025 reckoning says it rarely beats best-of-N at equal budget; conformity/drift (§3.8) | judge over proposals; **prefer tournament if any oracle exists** (P4/P5) |
| **Speculative exploration** | launch divergent approaches **in parallel** *before* committing (e.g. "try the trait refactor AND the enum refactor") → keep the one that verifies, discard the rest | exploratory forks where you'd otherwise serialize trial-and-error; **free locally** (cloud pays per fork) | launching forks you have no oracle to choose between → analysis paralysis | oracle-first selection; losers' worktrees pruned |

#### 4.2.2 The selection rule (normative)

```
choose_pattern(task) =
  if not can_write_acceptance_oracle(task):        # ch.02 §4.5 acceptance up front
      if task.needs_breadth_isolation:  PLANNER_WORKERS_MERGER   # research/investigation
      elif task.is_subjective_synthesis: DEBATE (fallback)       # last resort, no oracle
      else:                              SINGLE_AGENT             # the safe default
  else:                                                          # we HAVE an oracle ⇒ verify-select
      if partitions_into_disjoint_footprints(task): FAN_OUT_MAP_REDUCE
      elif has_clean_staged_handoffs(task):         PIPELINE
      elif one_hard_goal_high_failure_rate(task):   TOURNAMENT_BEST_OF_N
      elif exploratory_divergent_approaches(task):  SPECULATIVE_EXPLORATION
      else:                                         SINGLE_AGENT
  # in ALL cases, clamp fan-out width to Governor.max_concurrent_runs (P6)
```

**The decisive heuristic (P4/P5): the presence of a deterministic oracle flips the strategy from *coordinate* to *verify-and-select*.** When you can write `acceptance.oracle` (build+test+grep), the reliable move is to *generate many and let the oracle pick* — not to make agents agree. Debate is reserved for the genuinely oracle-less case, and even then is a documented fallback. This is the chapter's entire stance on the multi-agent reckoning (§3.8) made operational.

#### 4.2.3 Footprint analysis (what makes fan-out safe)

Before fanning out, `merge::plan` computes each subtask's **predicted file footprint** (from the plan's `produced`/target paths + a cheap static touch-set). Subtasks with **disjoint footprints** parallelize freely (no merge conflict possible by construction). Subtasks with **overlapping footprints** are either (a) **serialized** on a shared worktree (dependency edge added to the DAG), or (b) allowed to race **only** under tournament semantics (one winner adopted, others discarded). This **prevents** most conflicts rather than resolving them — the single biggest lever against §3.4's "merge is the hard part."

---

### 4.3 The isolation model

Parallel work is isolated at **four** levels; worktrees alone are insufficient (the §3.4 runtime-isolation gap). All of this **references ch.10's sandbox/capability model** for *what* a run may touch — `hide-fleet` owns the *orchestration* (lifecycle, leasing), ch.10 owns the *enforcement* (the sandbox boundary).

| Level | Mechanism | Prevents | Owner |
|---|---|---|---|
| **Filesystem** | **git worktree** per run: `git worktree add .hide/wt/<run_id> <base>` off the same `.git`. Effects land in the run's tree; the user's working tree is never touched by a parallel run (P3). Optional **overlay-FS** (copy-on-write) for non-git assets. | parallel edits trampling each other; a broken build in one run breaking the user's editor | `isolate/worktree.rs` |
| **Ports** | a **PortLease** allocator hands each run a disjoint port range from a pool; injected as env (`PORT`, `DATABASE_URL` host:port) so dev-servers/tests in different runs don't collide on 3000/5432/8080 (the named §3.4 gap) | "launch two React apps, one fails"; test DBs clobbering each other | `isolate/ports.rs` |
| **Process / env** | each run gets a namespaced **TMPDIR**, build cache dir, and **unique DB/schema names** (`hide_run_<id>`); shell tools run under ch.10's sandbox with the run's capability grant | shared scratch/cache corruption; one run's migration hitting another's DB | `isolate/env.rs` + **ch.10** |
| **Network** | inherited from **ch.10**: default deny-egress; per-run capability grant for any allowed host. Parallel runs cannot exfiltrate or interfere via the network beyond their grant | a swarm hammering an external API; cross-run network interference | **ch.10** (referenced) |

**Worktree lifecycle (pseudocode):**

```rust
fn isolate_run(run_id, base_ref, caps: CapabilityGrant /* from ch.10 */) -> RunWorkspace {
    let wt = git_worktree_add(repo, format!(".hide/wt/{run_id}"), base_ref)?;  // own dir, shared .git
    let ports = port_pool.lease(run_id, n_ports_for(caps))?;                   // disjoint range
    let env = env_seed(run_id, &wt, &ports);                                   // TMPDIR, DB names, caches
    let sandbox = ch10::open_sandbox(&wt, &caps, &env)?;                        // ch.10 enforces boundary
    emit("workspace.created", { run_id, path: wt, ports, base_ref });
    RunWorkspace { wt, ports, env, sandbox }
}
fn release_run(ws: RunWorkspace, outcome: RunOutcome) {
    // merge happens in §4.4 BEFORE release for an adopted run; here we just clean up.
    port_pool.release(ws.ports);
    if outcome.discarded { git_worktree_remove(ws.wt); }     // tournament losers / speculative discards
    else { git_worktree_prune_after_merge(ws.wt); }
    emit("workspace.released", { run_id: ws.id, kept: !outcome.discarded });
}
```

**Why worktrees over full clones or containers (on Apple Silicon).** Full clones duplicate the repo per run (RAM/disk waste at swarm scale); containers add per-run memory overhead that **competes with the model for unified memory** — the worst trade on a RAM-bound box (§3.8). Worktrees share the object store (cheap), give true file isolation, and leave RAM for the runtime. Containers remain an **opt-in** for runs needing full env isolation (a ch.10 capability), but are not the default precisely because of the unified-memory cost.

---

### 4.4 Merge & conflict resolution

The "N agents attempt → judge selects/merges" flow (P3/P4), built on ch.02's oracle gate. Two distinct cases: **fan-out** (disjoint footprints → combine all) and **tournament** (same goal → select one). Both funnel through an **integration branch** (§3.4's proven pattern).

#### 4.4.1 The integration-branch funnel (fan-out / map-reduce)

```
N worktree runs each pass their OWN acceptance oracle (ch.02 §4.6)  ──┐
                                                                       ▼
   create integration branch off base  (git worktree add .hide/wt/integ)
                                                                       │
   for each completed run, in footprint order (disjoint first):       │
       attempt merge run.branch → integ                               │
         ├─ clean  → keep                                             │
         └─ CONFLICT → §4.4.3 resolution (structured → 3-way → judge) │
                                                                       ▼
   run the FULL test suite on integ  (not just per-run subsets)       │
         ├─ green → promote: fast-forward/merge integ → user branch   │  (the ONLY effect-commit)
         └─ red   → bisect the offending run, drop or repair it,      │
                    re-integrate the rest                             │
                                                                       ▼
                              emit merge.completed { adopted[], dropped[], conflicts[] }
```

The funnel guarantees the user's branch only ever receives a **fully-integrated, full-suite-green** result — never a half-merged or individually-green-but-jointly-broken state. This is the structural answer to "8 agents each passed their own tests but together broke the build."

#### 4.4.2 The tournament selector (best-of-N / speculative)

```rust
/// N runs attempted the SAME goal in isolated trees. Pick the winner. (P4)
fn select_winner(cands: Vec<RunOutcome>, env) -> Selection {
    // 1. REGRESSION FILTER (the SWE-agent recipe, §3.2): drop any candidate that
    //    fails the existing suite — a fix that breaks other tests is disqualified.
    let viable: Vec<_> = cands.into_iter()
        .filter(|c| c.oracle.regression_clean())          // deterministic, authoritative
        .collect();
    if viable.is_empty() { return Selection::None; }       // → repair/replan (ch.02)
    // 2. ORACLE RANK among survivors: more acceptance-oracles passed, fewer warnings,
    //    smaller diff, faster runtime_smoke — all DETERMINISTIC signals.
    let mut ranked = rank_by_oracle_signals(viable);        // P4: oracles outrank everything
    // 3. JUDGE TIE-BREAK only among oracle-equivalent leaders (ch.02 §4.6.3):
    if ranked.leaders().len() > 1 {
        ranked = llm_judge_pairwise(ranked.leaders(), env); // position-swapped, rubric-grounded
    }                                                       // NEVER overrides an oracle (P4)
    let winner = ranked.first();
    emit("merge.selected", { winner: winner.run_id, beaten: ranked.rest_ids(), basis: ranked.basis });
    Selection::Adopt(winner)                                // → integration funnel (single merge)
}
```

**The selector is oracle-first by construction** (P4): a candidate that passes more deterministic oracles always outranks one a judge merely *prefers*. The judge runs only to break ties among candidates the oracles cannot separate, with ch.02's bias mitigations (pairwise, position-swapped, rubric-grounded). Losers' worktrees are pruned (§4.3). This is the §3.2 SWE-agent recipe — regression-filter → rank → vote — with the vote downgraded to a tie-break because we have a real oracle.

#### 4.4.3 Conflict resolution ladder

When a merge conflicts, resolve in escalating order (cheap→expensive):

1. **Structured / semantic merge** (preferred): use **tree-sitter** ASTs (ch.01 §3) to merge at the *declaration* level — two runs that added different functions to the same file merge cleanly even if line-adjacent (git's line-diff sees a conflict; the AST sees two independent additions). Resolves the common "both added a route/handler" case automatically.
2. **Git 3-way merge** (recursive strategy): for textual conflicts the AST can't classify, fall back to standard 3-way against the common ancestor.
3. **An LLM resolver run** (ch.02 run, oracle-gated): hand the conflict hunks + both intents to a fresh run whose `acceptance` is "merged file builds + both runs' tests pass." The *resolution is itself verified* — it's not trusted until the suite is green. This is the only place generation touches the merge, and it's gated like everything else (P4).
4. **Escalate to human** (the honest fallback): if (1–3) fail or the autonomy level forbids auto-resolution, emit `merge.conflict{needs_human}`, present both diffs + the conflict in ch.03's diff UI, and pause (ch.02 `Paused`). **No silent wrong merge** (P3).

Every step emits events (`merge.conflict`, `merge.resolved{by}`) so the timeline shows exactly how each conflict was settled — auditable and replayable (ch.01 T1).

---

### 4.5 The task queue & job schema

The queue is **a projection of the event log** (ch.01 T1/T2): jobs are created/updated by events (`job.enqueued`, `job.started`, `job.completed`), so the queue survives crashes and is rebuilt by replay (P8). It is **not** a separate authoritative store.

#### 4.5.1 The Job schema (normative — Appendix A.1)

```jsonc
{
  "job_id": "job_01H…",                 // ULID
  "kind": "agent_run",                  // agent_run | batch | test_shard | research | merge | custom
  "title": "Add JWT refresh-token support",
  "priority": "interactive",            // interactive > high > normal > batch > idle  (§4.6)
  "created_by": "user" ,                // user | agent (a run spawned this) | schedule (cron)
  "parent_job": null,                   // for fan-out children / batch members (the job DAG)
  "deps": [],                           // job-level DAG edges (this runs after these complete)
  "run_spec": {                         // what ch.02 run to launch (the black box, P2)
    "goal": "…",
    "profile": "careful-refactor",      // ch.01 §4.10 agent profile (model, tools, autonomy)
    "budget": { /* ch.02 Budget */ },   // per-run caps (steps/wallclock/etc.)
    "orchestration": "tournament",      // §4.2 pattern hint (single|fanout|pipeline|tournament|…)
    "fanout": { "width": 4, "select": "oracle_first" }   // tournament/best-of-N params
  },
  "isolation": "worktree",              // §4.3 (worktree | overlay | container | none)
  "base_ref": "main@abc123",            // the commit each worktree forks from
  "resource_hint": {                    // §4.6 admission inputs (advisory; Governor decides)
    "est_ram_mb": 1200, "needs_gpu": true, "est_wallclock_ms": 600000,
    "concurrency_class": "model"        // "model" = competes for runtime batch; "cpu_only" = doesn't
  },
  "status": "queued",                   // queued|admitted|running|paused|preempted|merging|done|failed|cancelled
  "schedule": null,                     // {at: "02:00", gate: ["idle","ac_power"]} for batch (§4.7)
  "attempts": 0, "max_attempts": 1,     // job-level retry (distinct from in-run repair)
  "created_at": "…", "admitted_at": null, "finished_at": null,
  "result_ref": null,                   // blob/event ref to the run's outcome summary
  "schema_version": 1
}
```

```rust
enum JobStatus { Queued, Admitted, Running, Paused, Preempted, Merging, Done, Failed, Cancelled }
enum Priority  { Interactive, High, Normal, Batch, Idle }   // strict ordering; fair-share WITHIN a class
enum ConcurrencyClass { Model,    // holds a runtime generation slot (bounded by max_batch_size)
                        CpuOnly } // tests/builds/grep — bounded by CPU/RAM, NOT the model batch
```

**The `concurrency_class` distinction is load-bearing (P1/P6).** A `Model`-class job consumes one of the runtime's `max_batch_size` generation slots — these are the scarce resource. A `CpuOnly` job (running the test suite, building, grepping) does **not** compete for the model and can run far more widely (bounded by cores/RAM). So the scheduler maintains **two pools** with different ceilings — you can have 32 test shards running while only 6 agents are generating. This is the key to high throughput on one box.

#### 4.5.2 The job DAG & ready-set

Jobs form a DAG over `deps` (distinct from the *intra-run* step DAG of ch.02 §4.5 — this is the *inter-run* DAG). `dag.rs` computes the **ready-set** via Kahn's algorithm (a job is ready when all `deps` are `Done`), exactly the §3.5 topological progression. Fan-out children share a `parent_job`; the parent's reduce/merge job `deps` on all children. Cycle detection rejects a cyclic job graph at enqueue.

#### 4.5.3 Admission (backpressure)

```rust
/// A job leaves Queued only when the Governor's envelope has room (P7). Bounded admission,
/// unbounded completion-reporting (§3.5). NEVER over-commit the machine.
fn try_admit(queue: &mut Queue, gov: &Governor) {
    for job in queue.ready_by_priority() {                  // priority then fair-share (§4.6)
        match gov.can_admit(&job.resource_hint) {           // RAM/GPU/thermal/slot check
            Admit::Yes(grant) => { queue.admit(job, grant); launch(job); }
            Admit::No(reason) => { /* leave queued; emit backpressure telemetry */ break; }
            Admit::Defer      => continue,                  // skip this one, try lower-cost jobs
        }
    }
}
```

Admission **blocks** (jobs stay `Queued`) rather than dropping or over-committing — the producer-slows policy of ch.01 §4.9, applied to job admission. The completion side (a finished run reporting its outcome) is never blocked, guaranteeing forward progress (§3.5).

---

### 4.6 The scheduler & the resource Governor

This is the chapter's systems core: **a machine-wide Governor that extends ch.02's per-run Governor from one run's budget to the whole box's physical envelope** (P1/P6). ch.02's Governor caps *one run's* steps/wallclock/effects; this Governor caps *how many runs the machine can physically sustain* — on **RAM, GPU/runtime-batch, and thermal**, the real Apple-Silicon constraints (§3.8).

#### 4.6.1 The resource envelope (Appendix A.2)

```rust
struct ResourceEnvelope {                 // the machine-wide ceilings (P1)
    // Hard physical ceilings (admission denied on breach):
    ram_headroom_mb_min:   u64,   // keep ≥ this much unified RAM free (default: 20% of total) — NEVER swap
    max_model_runs:        u32,   // concurrent Model-class runs ≤ runtime max_batch_size (read from serve)
    max_cpu_runs:          u32,   // concurrent CpuOnly jobs (default: physical_cores − 1)
    max_worktrees:         u32,   // live worktrees (disk + inode bound; default 32)
    max_ports_leased:      u32,   // port-pool size
    // Thermal governor (soft → throttles admission, does NOT abort running work):
    thermal_backoff:       ThermalPolicy,  // {warn_pct, throttle_pct} of a thermal proxy
    // Fairness:
    fair_share:            FairSharePolicy, // weights per priority class
    preempt:               PreemptPolicy,   // when interactive needs a slot held by batch
}

struct GovernorState {                    // live, sampled by ResourceProbe (~1 Hz)
    ram_free_mb: u64,            // from OS (vm_stat / mach)
    thermal_level: f32,          // 0..1 proxy: dec_tps_now / dec_tps_baseline (per /metrics) — a DROP signals throttle
    model_runs_live: u32, cpu_runs_live: u32, worktrees_live: u32, ports_leased: u32,
    dec_tps_ewma: f32,          // smoothed throughput (degradation detector)
}
```

**The thermal signal without a private API.** macOS does not hand apps a clean "you are throttling" bit, so the Governor uses a **throughput-derived proxy**: it tracks `dec_tps` from `/metrics` against a per-model baseline; a sustained **drop of >X%** (the §3.8 "−20–40% after 10–15 min" signature) is read as thermal/contention throttle and **reduces the admission ceiling** (stops admitting new Model-class runs, lets in-flight ones drain). This converts the cited physical reality into a closed-loop control input — *the runtime's own throughput number is the thermometer.* (A future hook could read SMC/`powermetrics`; the proxy needs no privilege.)

#### 4.6.2 The scheduler loop (pseudocode)

```rust
/// Runs ~1 Hz (and on every job-state event). Priority → fair-share → admit under the envelope.
fn schedule_tick(queue: &mut Queue, gov: &mut Governor) {
    gov.refresh(&resource_probe());                         // sample RAM/thermal/dec_tps/live-counts

    // 1. THERMAL/RAM gate: if degraded, shrink the admission ceiling (don't kill running work).
    let ceiling = gov.effective_ceiling();                  // ↓ when ram low or dec_tps dropped (§4.6.1)

    // 2. PREEMPTION: if an INTERACTIVE job is queued and all model slots are held by BATCH/IDLE runs,
    //    preempt the lowest-priority batch run (checkpoint-and-yield, NOT kill — ch.02 §4.4 abort+ckpt).
    if queue.has_waiting(Interactive) && gov.model_slots_full() {
        if let Some(victim) = queue.lowest_priority_running(below = High) {
            preempt::checkpoint_and_yield(victim);          // flips GenerateRequest.abort; state durable
            emit("job.preempted", { victim: victim.id, for: "interactive" });
        }
    }

    // 3. ADMIT in priority order, fair-share within a class, under the (possibly-shrunk) ceiling.
    for job in queue.ready_ordered(by = (priority_desc, fair_share_within_class)) {
        if !ceiling.has_room_for(&job) { continue; }        // try cheaper jobs (CpuOnly may fit when Model can't)
        match gov.can_admit(&job.resource_hint) {
            Admit::Yes(grant) => { queue.admit(job, grant); spawn_run(job); }
            _ => {}                                          // stay queued (backpressure, P7)
        }
    }

    // 4. RESUME preempted jobs when room reappears (their state is durable — §4.7).
    for j in queue.preempted_ready() {
        if ceiling.has_room_for(&j) { resume_run(j); emit("job.resumed", { job: j.id }); }
    }
}
```

#### 4.6.3 Priority, fair-share, preemption

- **Priority classes (strict order):** `Interactive` (the user's foreground turn — *always* wins, P9) > `High` (user-initiated background task) > `Normal` (agent-spawned fan-out children) > `Batch` (overnight jobs) > `Idle` (opportunistic, e.g. skill-library gardening, speculative pre-warm). Interactive jobs **preempt** lower classes for a model slot.
- **Fair-share *within* a class** (the §3.5 weighted-fair-share): among equal-priority jobs, dispatch by least-recently-served + a per-originating-session weight, so one session's 8-way fan-out can't starve another session's runs. Prevents the "one user monopolizes the box" failure even single-user (one *session's* swarm vs another's).
- **Preemption = checkpoint-and-yield, never kill** (P8): a preempted Model-class run flips its `GenerateRequest.abort` at a token boundary (ch.02 §4.4), checkpoints its `AgentState` (ch.02 §4.13 — durable on the log), releases its slot, and goes `Preempted`. When room reappears, it **resumes from the checkpoint** — no lost work. This is the Kubernetes preempt-lower-priority idea (§3.5) made *lossless* by ch.02's resumability.

#### 4.6.4 The Governor as circuit-breaker (runaway-swarm safety, §3.8)

The same Governor *is* the cost-circuit-breaker (the local analogue of cloud spend caps): it enforces **max concurrent runs**, **max fan-out width per request**, **max worktrees**, and a **swarm wall-clock**. It also runs **anomaly detection** on the fleet: an EWMA on jobs-spawned-per-minute trips a breaker if an agent spawns runs faster than a threshold (the §3.8 "rapid increase in inter-agent transaction frequency" trigger) — a planner stuck in a fan-out loop is caught and paused, not allowed to fork the machine to death (P6). **Crucially, the breaker bounds *spawning*, never starves *running* work** — a tripped breaker stops *new* admissions and surfaces a banner; in-flight runs drain normally.

---

### 4.7 Overnight / batch jobs

The flagship local superpower: **start a swarm at midnight, wake to a report** (P1/P8/P12). A `BatchJob` is a job DAG (§4.5) with a `schedule` gate and a wake report.

#### 4.7.1 Batch lifecycle

```
user defines a BATCH (a queue of goals, e.g. "fix all 14 failing tests", "add types to /core",
   "try 3 refactors of the parser")  →  job DAG with schedule { at:"02:00", gate:["idle","ac_power"] }
        │
   SCHEDULE GATE fires (time reached AND machine idle AND on AC power — §4.7.2)
        │
   scheduler drains the DAG under the Governor envelope (§4.6) — fan-out under RAM/thermal caps,
   interactive-empty so the WHOLE machine is available; thermal backoff paces it across hours
        │
   each member run is CHECKPOINTED (ch.02 §4.13) — a reboot mid-batch resumes incomplete members
        │
   on completion (or morning): assemble the WAKE REPORT (§4.7.3) and (if configured) push a
   notification; results sit in worktrees/integration branches AWAITING REVIEW (not auto-merged
   to main unless autonomy == autonomous AND full suite green — ch.02 §4.3 autonomy)
```

#### 4.7.2 Schedule gates (don't cook the laptop)

A batch fires only when **all** its `gate` conditions hold — designed around the Apple-Silicon thermal reality (§3.8):

- **`idle`** — no interactive session active (the user isn't working) → the whole machine is the swarm's.
- **`ac_power`** — on AC, not battery (a Mac Studio is always-on; a laptop shouldn't drain/heat on battery).
- **`thermal_ok`** — the thermal proxy (§4.6.1) is nominal before starting; the run *paces* itself via thermal backoff thereafter (admit-fewer when hot, never abort).
- **`cron`** — a time window (`02:00–06:00`), implemented via the host's scheduler (or, for a Mac-Studio server, a launchd/cron trigger that pokes the fleet endpoint).

This is the "24/7 server" story made safe: a Mac Studio runs batches continuously within thermal limits; a laptop runs them only when plugged in, idle, and cool.

#### 4.7.3 The wake report (Appendix A.3)

```jsonc
{
  "batch_id": "batch_01H…",
  "ran": "2026-06-24T02:00 → 05:42",
  "summary": { "goals": 14, "succeeded": 11, "partial": 2, "failed": 1 },
  "results": [
    { "goal": "fix test auth::refresh", "status": "done",
      "outcome": "patch on integ branch, full suite green", "review_ref": "diff_…",
      "resource": { "wallclock_ms": 412000, "peak_ram_mb": 1300, "dec_tps_avg": 38 } },
    { "goal": "add types to core/parser", "status": "partial",
      "outcome": "9/12 files typed; 3 need human (ambiguous generics)", "review_ref": "diff_…" },
    { "goal": "refactor parser (3 approaches raced)", "status": "done",
      "outcome": "trait-based approach won (smallest diff, all tests pass); 2 alts discarded",
      "selection_basis": "oracle: -40% LoC, +0 test failures", "review_ref": "diff_…" }
  ],
  "needs_review": ["diff_…", "diff_…"],            // queued in ch.03's review surface
  "thermal_events": 2,                              // times the Governor backed off
  "total_runs": 31, "total_model_seconds": 9400
}
```

The report is a **projection over the batch's events** (P12) — every line is reconstructable from the log. It lands in ch.03's review queue so the morning ritual is "review 11 green diffs, handle 3 flagged ambiguities" — the user reviews *outcomes*, not process. **This is the product: you delegated 14 tasks to a free overnight team and woke to reviewable results.**

---

### 4.8 Parallel testing & parallel research

Two high-value `CpuOnly`/breadth applications that don't compete for model slots (§4.5.1) — so they run *wide*.

**Parallel testing (`test_shard` jobs).** The test suite is **sharded** across N `CpuOnly` jobs (by test file/module), each in its own worktree (so a flaky test mutating shared state can't poison siblings — the §4.3 isolation). Shards run bounded by cores/RAM, not the model batch, so a 32-core box runs ~31 shards at once. Results fan-in to a `test.status` aggregate (ch.01 §4.6). Used both *within* a run's `VERIFY` (ch.02 §4.6 — parallelize the oracle) and *standalone* ("run the whole suite 10× to catch flakes"). **[PROVEN]** (sharded CI is universal), difficulty *low*, impact *high* (verification is the bottleneck; parallelizing it speeds every run).

**Parallel research (the orchestrator-worker pattern, §3.3).** A research goal ("understand how auth flows through this codebase", "survey 5 libraries for X") fans out to subagents (ch.02 §4.10, `kind: research`, `isolation: context`) each with **its own context window** exploring one facet, returning a tight summary (the `return_contract`). A merger run synthesizes + de-duplicates. This is Anthropic's 90.2% architecture — and the **15× token cost that gates it for cloud is free for us** (P1), so HIDE can run *deeper* research fan-outs than a metered cloud agent would dare. **[PROVEN]** (Anthropic shipped it), difficulty *low* (ch.02 subagent contract exists), impact *high* for breadth tasks.

---

### 4.9 Workstation / remote mode

**[TIER-3: WORKSTATION]** The headline workstation story: **a laptop thin-client drives a Mac-Studio agent server.** The Mac Studio (big unified memory, always on AC, thermally generous) runs the runtime + `hide-fleet`; the laptop runs a thin HIDE client that submits intents and renders events. The server is **authoritative**; the client is a disposable view (P10) — exactly ch.01's intent-in/events-out model, now over a network.

#### 4.9.1 Design stance (cited)

The protocol is **ACP-shaped** (§3.7): JSON-RPC 2.0, session-centric, resumable — so a HIDE client could interoperate with other ACP agents and vice-versa — but the **update payload is the ch.01 Event envelope** (richer than ACP's notifications: carries `seq`/`run_id`/`parent`/`cause`, replay-safe). Reconnection follows the §3.6/§3.7 durable-session model: **server-side sessions outlive the socket; reconnect resumes from `seq` with no duplicate events.** Transport and auth follow the Codex/OpenClaw posture (§3.7): **server keeps working when the client disconnects; resync on reconnect; wss/TLS or SSH-tunnel; loopback by default** (P11).

#### 4.9.2 The wire protocol (normative — Appendix A.4)

A persistent **WebSocket (wss)** carrying JSON-RPC 2.0, chosen over plain SSE/HTTP because remote agent control is **bidirectional and long-lived** (the §3.7 "why WebSockets not HTTP" lesson: intents up, events down, approvals up, all on one resumable connection). Three message classes:

```jsonc
// 1. CLIENT → SERVER : intents (requests to do something — ch.01 Wire A, over the wire)
{ "jsonrpc":"2.0", "id": 42, "method": "hide/intent",
  "params": { "intent": "SubmitTurn", "session": "sess_…", "body": { "text":"add JWT refresh" } } }
// server acks immediately (ch.01 ack-then-events): { "jsonrpc":"2.0","id":42,"result":{"accepted":true,"event_seq":1043} }

// 2. SERVER → CLIENT : event stream (the projection — ch.01 Event envelope, verbatim)
{ "jsonrpc":"2.0", "method": "hide/event",
  "params": { /* the ch.01 Event: seq, id, run_id, parent, cause, kind, payload, … */ } }

// 3. RECONNECTION : client resumes from last seq it durably saw (§3.6 durable session)
{ "jsonrpc":"2.0", "id": 1, "method": "session/resume",
  "params": { "session":"sess_…", "from_seq": 1043, "client_token":"…" } }
// server replays (from_seq, head] from the log — NO duplicate events, NO re-fired effects (ch.01 T3)
```

**Session model.** `session/new` opens a server-side session (optionally declaring MCP servers in the handshake, ACP-style); `session/resume{from_seq}` reattaches after a drop and replays the gap; the session **persists on the server independent of the connection** (§3.6) — a batch the laptop kicked off keeps running with the laptop *asleep*, and the laptop resyncs on wake. Approvals (ch.02 `Paused`/autonomy) round-trip as intents (`hide/intent{ApproveDiff}`), so a `suggest-only` remote run pauses for the laptop user exactly as a local one would.

#### 4.9.3 Auth & security posture (REFERENCES ch.10)

**Deny-first (P11).** The agent server binds **loopback only** by default; remote exposure is an explicit opt-in with three supported modes, in order of preference:

1. **SSH tunnel (preferred):** the laptop port-forwards to the Mac Studio's loopback endpoint over SSH; the WS stays `ws://localhost:<fwd>` end-to-end, auth tokens never traverse a raw network (the §3.7 "loopback + SSH port-forward preferred" rule). Zero new attack surface beyond SSH.
2. **wss + token (LAN):** TLS-terminated WebSocket on the LAN, bearer token from a **device-pairing** handshake (`pair.rs`: scan a QR / enter a 6-digit code shown on the server → mints a scoped client token stored in the laptop's trust store). Tokens sent **only** over `wss://` (§3.7).
3. **wss + mTLS (hardened):** mutual-TLS with client certs for a fixed, trusted laptop ↔ Studio pair.

**Capability scoping is ch.10's job, referenced here:** a remote client's token carries a **ch.10 capability grant** — what that client may make the server do (e.g. a read-only "monitor" client that can watch the fleet but not approve diffs or run shell). `hide-fleet` *transports and enforces presence of* the grant; **ch.10 defines and validates the grant itself.** The remote server never grants ambient authority (ch.01 T4); every remote intent is checked against the session's capability set before it appends an event. **No remote intent can exceed what a local user of that profile could do.**

#### 4.9.4 State sync, streaming, reconnection (the reliability core)

- **Server-authoritative.** All state lives in the server's event log (ch.01 T1). The client holds only a *projection* it can rebuild — so a client crash/reload loses nothing (P10), identical to ch.01's disposable-WebView guarantee, extended over the wire.
- **Streaming.** Token streams (40–120 tok/s) ride the WS as coalesced `token_batch` events (ch.01 §4.4 render-coalescing) so a slow/remote link gets batched updates, not per-token floods.
- **Reconnection.** On drop, the client reconnects and sends `session/resume{from_seq}`; the server replays `(from_seq, head]` from the log — **exactly-once delivery by construction** (events are immutable and `seq`-ordered; replaying past events re-applies recorded data, never re-fires effects — ch.01 T3). A 3-hour batch that the laptop slept through replays as a fast fold on reconnect.
- **Backpressure over the wire.** The server's projection→socket channel is bounded and coalesces under a slow link (ch.01 §4.9); the log keeps every event, the wire gets a paced summary.

---

### 4.10 Cross-machine distribution (optional)

**[TIER-4: CLUSTER]** **[SPECULATIVE]**, difficulty *high*, impact *medium* (most users have one box; this is for a power user with several Macs). The single-box scheduler (§4.6) generalizes to a **node pool**: each node advertises its `ResourceEnvelope` (RAM, `max_batch_size`, thermal headroom); the scheduler dispatches jobs cross-box by the same priority/fair-share/admission logic, choosing the node with the best fit. This is the §3.2 "Kubernetes schedules containers across a node pool" idea, scaled *down* to a few trusted home machines.

**Design constraints that make it tractable (not a distributed-systems quagmire):**
- **The event log stays single-authority per session** — a session is pinned to one node; cross-node is *job-level* distribution, not *event-level* consensus (no distributed log, no Raft). Each job runs wholly on one node and reports its outcome back to the originating node's log.
- **Worktree/merge stays local to the executing node** — a job's isolation and oracle run where it runs; only the *result* (a diff + verdict, content-addressed) crosses the network, merged on the originating node via §4.4.
- **Transport reuses §4.9** — nodes speak the same wss/JSON-RPC + capability-scoped protocol to each other (a node is just a trusted "client" that also accepts jobs).
- **Failure = job-level retry** — a node dropping mid-job is the §4.6 preempt-and-resume path, re-dispatched to another node from the last checkpoint (the job's `run_spec` + checkpoint is portable).

This stays *optional* and *post-everything*: the single-box workstation mode (§4.9) is the 95% story; the cluster is a moonshot for the user who already owns a Mac Studio *and* a Mac mini *and* a laptop and wants them to act as one.

---

## 5. How we EXCEED ("cloud literally cannot do this")

Each claim ties to a tenet and contrasts the structural cloud limitation.

| # | HIDE does | Cloud structurally cannot | Why (the moat) |
|---|---|---|---|
| **1** | **Run dozens of agents in parallel for $0.** Best-of-N at width 8, fan-out across 12 files, 31 test shards — no bill, no rate limit. | Cloud charges **per agent-hour / per token**; width-8 best-of-N is 8× the bill, so cloud users ration it (the §3.2 recipe samples *5–10* and stops). | **P1: tokens are free; the machine is the budget.** The §3.2 SWE-agent recipe that cloud users *meter* is our *default*. |
| **2** | **24/7 workstation server.** A Mac Studio runs swarms continuously; the laptop connects/disconnects at will and the work persists (§4.9). | Cloud agents are **session-scoped and metered**; "always-on" means "always-billing." There is no free idle capacity. | **P10/P1.** Your hardware's idle time is free; cloud's idle time is someone else's revenue. |
| **3** | **Overnight swarms with wake reports** (§4.7). Delegate 14 goals at midnight, wake to 11 reviewable green diffs. | An overnight cloud swarm is a **large overnight bill** with no upper bound on cost; users won't leave it running. | **P1/P8/P12.** Free compute + durable resumable batch + thermal-gated pacing = unattended overnight is *safe and free*. |
| **4** | **Your Mac Studio is the cluster** (§4.10). A few home Macs act as one private agent fleet, no per-node cloud cost. | Cloud parallelism is **rented**; scaling out is scaling the bill. | **P1.** Owned silicon amortizes to zero marginal cost; rented silicon never does. |
| **5** | **Oracle-first selection on free compute** (§4.4). N attempts, the *compiler/test* picks the winner — and we can afford a *large* N. | Cloud best-of-N is gated by per-trajectory cost, so N stays small and the selector is often a *cheap judge* not a *full regression run*. | **P4.** We run the *full* regression filter on *every* candidate because compute is free — the selector is ground truth, not a budget-constrained heuristic. |
| **6** | **Deeper research fan-outs** (§4.8). Anthropic's orchestrator-worker (90.2%) at *15× tokens* — free for us, so we fan out *wider/deeper* than a metered agent. | The **15× token multiplier** is real money in the cloud; users cap research breadth to control spend. | **P1.** The exact cost multiplier that gates cloud research is zero for us. |
| **7** | **Determinism + durable event log across all of it** (§4.5/§4.9). Every parallel run is replayable, every batch resumable, every remote session reconnectable — byte-reproducible where greedy. | Cloud sessions are **ephemeral and opaque**; you can't replay a competitor's overnight swarm or reconnect to a dropped 3-hour batch from a different device. | **ch.01 T1/T2/T6 + P8/P10.** The log is the user's file; the fabric is a projection over it. |
| **8** | **Lossless preemption** (§4.6). Interactive always wins a model slot; the batch run yields *with a checkpoint* and resumes — no lost work. | Cloud preemption (spot instances) typically **kills** the job; resumability is the customer's problem. | **P9/P8 + ch.02 §4.13.** ch.02's resumability makes Kubernetes-style preemption *lossless*, not destructive. |

> **The one-sentence moat.** *Cloud coding agents are throttled by a per-agent-hour meter; HIDE is throttled only by your Mac's RAM and heat — so the strategies the literature proves work (best-of-N with a verifier, orchestrator-worker research, overnight swarms) and that cloud users must ration, HIDE runs by default, for free, all night, from your couch.*

---

## 6. Failure modes + mitigations

The four named risks (deadlock, thrash, merge conflicts, runaway swarms) plus the MAST taxonomy (§3.8) mapped to mitigations. Each cites the mechanism that handles it.

| # | Failure | Cause | Mitigation (mechanism) |
|---|---|---|---|
| **F1** | **Deadlock** (job A waits on B, B on A; or all slots held by jobs waiting on slots) | a cyclic job DAG; or every model slot held by a run that spawned a child needing a slot | **Cycle detection at enqueue** (`dag.rs`, rejects cyclic graphs); **`max_stack_depth`** (ch.02 K8) bounds spawn recursion so a parent can't deadlock on a grandchild; **preemption** (§4.6.3) frees a slot for a starved interactive job. A child needing a slot its parent holds is detected (the parent is *waiting*, not *running*) and the child is admitted by yielding the parent. |
| **F2** | **Thrash** (machine swaps; thermal throttle; tok/s collapses across all runs) | over-admission past RAM headroom; too many Model-class runs; sustained heat | **The Governor admits on RAM headroom + `max_batch_size` + thermal proxy** (§4.6.1) and **never lets RAM-free drop below the floor** (no swap, P1); **thermal backoff** shrinks the ceiling when `dec_tps` drops (the §3.8 throttle signature); **two pools** (Model vs CpuOnly) keep test/build width from stealing model RAM. |
| **F3** | **Merge conflicts / jointly-broken integration** (N runs each green, together broken) | overlapping footprints; emergent interaction between independently-correct patches | **Footprint-disjoint scheduling** *prevents* most (§4.2.3 — overlapping footprints serialize or tournament); **the integration-branch funnel runs the FULL suite** before promoting (§4.4.1 — catches joint breakage); **structured (AST) → 3-way → verified-LLM-resolver → human** ladder (§4.4.3); **no silent wrong merge** — unresolved conflicts pause for human (P3). |
| **F4** | **Runaway swarm** (an agent spawns runs in a loop; the box forks to death) | a planner stuck fanning out; a bug spawning children unboundedly | **The Governor is the circuit-breaker** (§4.6.4): `max_concurrent_runs`, `max_fanout_width`, **EWMA on spawn-rate** trips a breaker (the §3.8 trigger); **bounds spawning, never starves running work**; surfaces a banner + pauses the offending planner. Spawn recursion bounded by `max_stack_depth` + `max_subagents` (ch.02 K8). |
| **F5** | **MAST: spec/design failures (41.8%)** — subagents duplicate work or leave gaps | vague subtask objectives; over-decomposition | **ch.02's `SubagentSpec.return_contract`** forces crisp objective/boundaries/output-format per subagent (the Anthropic lesson, §3.3); **the selection rule (§4.2.2) defaults to single-agent** when a task isn't cleanly separable — over-decomposition is *prevented*, not patched. |
| **F6** | **MAST: inter-agent misalignment (36.9%)** — agents drift, lose the goal, conform | debate/consensus dynamics; over-exploration (§3.8) | **Prefer oracle-first selection over debate** (P4) — there's no consensus to conform to, the *oracle* decides; **debate is a documented fallback only** (§4.2.1); each run has its own pinned goal + plan (ch.02 §4.5), so drift is bounded per-run. |
| **F7** | **MAST: verification gaps (21.3%)** — no one checks the combined result | per-part verification without whole verification | **The integration funnel's full-suite gate** (§4.4.1) is the whole-result oracle; **every job declares acceptance up front** (ch.02 §4.5); the merger/reduce step is itself a ch.02 run with its own oracle. The §3.8 "21% verification gap" is structurally closed by the funnel. |
| **F8** | **Remote: dropped connection mid-batch; stale client; token leak** | network flakiness; client crash; insecure exposure | **Server-authoritative + reconnect-resume from `seq`** (§4.9.4 — no lost work, exactly-once); **loopback-default + SSH/wss + capability-scoped tokens** (§4.9.3, ch.10); **tokens only over wss/loopback** (§3.7); a stale client just resyncs (it holds no authority, P10). |
| **F9** | **Worktree/port exhaustion; orphaned worktrees** | a crash leaving worktrees; port-pool drained | **`max_worktrees`/`max_ports_leased` ceilings** (§4.6.1); **worktree GC on boot** (prune orphans, ch.01 `tmp/` cleared on boot); port leases released on run end or reclaimed on crash-recovery replay (the dangling-Action detection, ch.01 §4.12). |
| **F10** | **Battery drain / overheating on a laptop** (batch cooks the machine) | running a swarm unplugged or while in use | **Schedule gates** `ac_power` + `idle` + `thermal_ok` (§4.7.2) — a laptop runs batches *only* plugged in, idle, and cool; the Mac Studio (always AC) has no such restriction. Thermal backoff paces any running batch (F2). |

---

## 7. Extensibility (new orchestration patterns)

Per ch.01's mandate ("to add capability X, does anyone touch `core/`?"), new orchestration patterns and scheduler policies are **plugins**, not core edits.

- **New `Pattern`** (a `pattern-provider` extension kind): implements the `Pattern` trait (`fan_out → runs → reduce`), registers via the ch.01 manifest (§7.2), and is selectable by name in a job's `orchestration` field. A plugin could add **map-reduce-with-PRM-reranking**, **iterative-refinement-tournament** (winner seeds the next round), or **diversity-forced best-of-N** (different temperatures/profiles per branch) without touching `hide-fleet` core. The selection rule (§4.2.2) is itself a registered policy a plugin can override.
- **New scheduler policy** (a `scheduler-policy` extension): a plugin supplies an alternative `ready_ordered`/`can_admit` (e.g. a deadline-EDF policy, or a GPU-aware policy when a future runtime exposes per-run VRAM). The Governor's `ResourceEnvelope` is config (ch.01 §4.10), so ceilings are tunable per-profile without code.
- **New `ConcurrencyClass`** — the Model/CpuOnly split is extensible; a plugin runtime that exposes a *third* resource (e.g. a separate embedding model server) registers a class with its own ceiling.
- **New isolation backend** (an `isolation-provider`): worktree is the default; a plugin can add **container-use** (full env isolation, §3.4) or **microVM** isolation as a selectable `isolation` value, gated by a ch.10 capability.
- **New remote transport** (a `remote-transport` extension): wss/JSON-RPC is default; a plugin could add a **QUIC** transport or a **relay-server** mode (for NAT traversal without SSH) — all carrying the same ch.01 Event payload, all gated by ch.10 auth.
- **New schedule trigger** (a `schedule-trigger` extension): `cron`/`idle`/`ac_power` are built-in; a plugin could add **git-hook triggers** ("run the test swarm on every push") or **file-watch triggers** ("re-run affected shards when `src/` changes").

Every pattern/policy a plugin contributes is still clamped by the Governor (P6) and emits the same `job.*`/`merge.*` events (ch.01 T1) — extensibility never escapes the safety chokepoint or the audit log.

---

## 8. Bleeding-edge / moonshots (ranked)

Ranked by *(impact ÷ difficulty)*; each tagged.

1. **Overnight test-flake hunter + auto-bisect.** **[RESEARCH-PROVEN orchestration; difficulty med; impact high].** A nightly batch runs the suite 50× across sharded worktrees (free, §4.8), clusters failures, and for each flake spawns a tournament of fix-attempts (§4.4.2) gated on "fails 0/50 after the fix." Wake to a list of *fixed* flakes + bisected root commits. Pure composition of §4.4/§4.7/§4.8 — buildable now once the tier lands.
2. **Speculative pre-warm of the next step.** **[SPECULATIVE; difficulty med; impact med-high].** While the user reviews step *k*'s diff, an `Idle`-class run speculatively executes the *predicted* step *k+1* in a throwaway worktree (free compute, P1). If the user accepts *k* and *k+1* matches the prediction, it's already done; else discard. Cloud can't afford speculative-execute-and-discard; we can. Needs a step-predictor + careful discard discipline.
3. **Diversity-forced best-of-N.** **[RESEARCH-PROVEN; difficulty low; impact med].** Instead of N identical-temperature samples, fan out N branches with *deliberately diverse* configs (different profiles/temperatures/approaches in the prompt) so the oracle selects from a genuinely varied pool — the §3.2 recipe sharpened. Low difficulty (a `Pattern` plugin, §7), measurable impact on hard-goal success.
4. **Cross-machine home cluster (§4.10).** **[SPECULATIVE; difficulty high; impact med].** Several home Macs as one fleet. High difficulty (the §4.10 constraints keep it tractable but it's still distribution); medium impact (few users have the hardware). Genuinely "your own cluster" — a flagship moonshot for the power user.
5. **Learned scheduler (RL over the resource envelope).** **[SPECULATIVE; difficulty high; impact med].** Replace the hand-tuned admission heuristic with a policy learned from the box's own telemetry (RAM/thermal/dec_tps history) to maximize throughput-under-thermal. The §3.8 "latency-aware orchestration" direction. Risky (a learned scheduler can misbehave); the hand-tuned Governor (§4.6) is the safe baseline it must beat.
6. **PRM-guided reduce.** **[RESEARCH-PROVEN; difficulty high; impact med].** Use a small trained process-reward model (ch.02 §8) to rank *trajectories* in the tournament reduce, not just final patches — catch a candidate that passed by luck via a suspicious path. High difficulty (needs the PRM); medium incremental impact over the oracle-first selector.

---

## 9. Open questions / dials

| # | Question / dial | Default | Trade |
|---|---|---|---|
| **Q1** | **`max_concurrent_model_runs`** — how many generating agents at once? | `= runtime max_batch_size` (read live) | Higher = more parallelism but shares fixed GPU compute (§3.8 "two queries share, don't add tok/s") → diminishing returns + RAM pressure. The runtime's batch size is the principled ceiling; exceeding it queues. |
| **Q2** | **RAM headroom floor** before refusing admission | 20% of total unified RAM | Lower = more concurrency, higher swap risk (catastrophic, P1). Conservative default; a Mac-Studio-with-192GB user can lower it. |
| **Q3** | **Thermal proxy threshold** (dec_tps drop → back off) | back off at >25% sustained drop | Too sensitive = under-utilizes a cool machine; too lax = thermal thrash. The proxy is indirect (§4.6.1); a future `powermetrics` hook would sharpen it. **Open: is a throughput-derived proxy reliable enough, or do we need the privileged SMC read?** |
| **Q4** | **Default fan-out width** for tournament/best-of-N | 4 | Higher = better hard-goal success (§3.2 uses 5–10) but more RAM/time; clamped by Q1 regardless. Profile-tunable. |
| **Q5** | **Auto-merge autonomy for batch** — promote green results to `main` unattended? | **No** (land on integration branch, queue for review) | `autonomous` overnight that auto-merges full-suite-green results is *possible* (ch.02 §4.3) and maximally hands-off, but riskier; default is review-on-wake. The user opts into auto-merge per-batch. |
| **Q6** | **Remote default exposure** | **loopback only** (SSH-tunnel to go remote) | wss-on-LAN is more convenient but more surface (P11); loopback+SSH is the secure default. **Open: ship a built-in relay for NAT traversal, or keep it SSH-only?** (a relay is convenient but is new trusted infra — §7 leaves it to a plugin.) |
| **Q7** | **Footprint prediction accuracy** — how good is the static touch-set that drives disjoint scheduling (§4.2.3)? | conservative (over-estimate footprint → over-serialize) | Over-estimating footprints serializes safely but loses parallelism; under-estimating risks merge conflicts. **Open: how much does a cheap static analysis buy vs just letting the integration funnel catch conflicts?** |
| **Q8** | **Preemption granularity** — checkpoint at any token boundary, or only at step boundaries? | step boundary (ch.02 transition) | Token-boundary preemption is faster to free a slot but checkpoints mid-step; step-boundary is cleaner but slower to yield. ch.02's one-transition-per-call makes step-boundary cheap; default there. |
| **Q9** | **`CpuOnly` width** (test/build shards) | `physical_cores − 1` | Higher saturates cores (good for throughput) but competes with the runtime's CPU-side work + the WebView. Leaves headroom by default. |

---

## 10. Cross-references

- **ch.01 (System Architecture)** — **binds hard.** Every parallel run emits the **Event envelope** (§4.6: `seq` ordering authority, `run_id`/`parent`/`cause` for the run tree, Action/Observation classing, replay-never-re-fires-effects). The queue/batch/remote layers are **projections of the event log** (T1/T2); jobs/merges/remote-sessions emit **new registered kinds** (`job.*`, `merge.*`, `workspace.*`, `remote.*`) via the manifest (§7.2). Backpressure (T8), config layering (§4.10), and the disposable-view model (T2) are inherited. The runtime sidecar's `max_batch_size`/`/metrics` (§4.3) feed the Governor.
- **ch.02 (Agent Kernel)** — **binds hard.** A scheduled job *is* a ch.02 run (P2). We **extend** its per-run Governor (§4.3) to a machine-wide one (§4.6) without contradiction. We **consume** the `SubagentSpec`/`SubagentReturn` contract (§4.10) for fan-out/research, the **best-of-N + oracle-selection** centerpiece (§4.8) as our tournament (§4.4.2), the **autonomy levels** (§4.3) for batch auto-merge policy, **checkpoint/replay** (§4.13) for lossless preemption + resumable batch, and the **`acceptance` oracle** (§4.5/§4.6) as the merge gate. The **selection rule (§4.2.2)** is the multi-run extension of K12 ("single-agent is the default").
- **ch.10 (Local-First Security)** — **references, does not duplicate.** ch.10 owns the **sandbox + capability model**; this chapter's **isolation** (§4.3 — worktree/port/env *orchestration*) sits on top of ch.10's *enforcement*, and the **remote protocol** (§4.9.3 — auth/token/mTLS) carries and checks ch.10 **capability grants** but never defines them. No remote intent exceeds the local profile's capabilities; the server grants no ambient authority (ch.01 T4).
- **ch.03 (Editor)** — consumes our `merge.*` and the **wake report** (§4.7.3): conflicts render in the diff UI (§4.4.3 step 4), batch results land in a review queue, the **fleet view** (§5/§4.1 `fleetview.rs`) is a panel projection.
- **ch.04 (Context/Memory)** — subagents get minimal **context seeds** (§4.8 research fan-out); parallel-research summaries fold into memory; the skill library's idle-time gardening is an `Idle`-class job (§4.6.3).

---

## Appendix A — Binding contracts

> These are the schemas **other chapters import**. Additive-by-default (ch.01 T10); unknown fields survive round-trips.

### A.1 — `Job` / `JobSpec` (the queue unit) — §4.5.1

```jsonc
Job {
  job_id: ULID, kind: "agent_run"|"batch"|"test_shard"|"research"|"merge"|"custom",
  title: string, priority: "interactive"|"high"|"normal"|"batch"|"idle",
  created_by: "user"|"agent"|"schedule", parent_job: ULID?, deps: ULID[],
  run_spec: { goal: string, profile: string, budget: ch02.Budget,
              orchestration: "single"|"fanout"|"map_reduce"|"pipeline"|"tournament"|"debate"|"speculative",
              fanout: { width: u8, select: "oracle_first"|"judge"|"vote" }? },
  isolation: "worktree"|"overlay"|"container"|"none", base_ref: string,
  resource_hint: { est_ram_mb: u64, needs_gpu: bool, est_wallclock_ms: u64,
                   concurrency_class: "model"|"cpu_only" },
  status: "queued"|"admitted"|"running"|"paused"|"preempted"|"merging"|"done"|"failed"|"cancelled",
  schedule: { at: string, gate: ("idle"|"ac_power"|"thermal_ok"|"cron")[] }?,
  attempts: u32, max_attempts: u32,
  created_at, admitted_at?, finished_at?, result_ref: BlobRef?, schema_version: 1
}
```

### A.2 — `ResourceEnvelope` / `GovernorState` (the machine-wide budget) — §4.6.1

```jsonc
ResourceEnvelope {
  ram_headroom_mb_min: u64,      // never admit below this free-RAM floor (no swap)
  max_model_runs: u32,           // ≤ runtime max_batch_size
  max_cpu_runs: u32, max_worktrees: u32, max_ports_leased: u32,
  thermal_backoff: { warn_pct: f32, throttle_pct: f32 },   // dec_tps-drop thresholds
  fair_share: { weights: { interactive, high, normal, batch, idle } },
  preempt: { enabled: bool, min_class_to_preempt: Priority }
}
GovernorState {  // live, ~1 Hz
  ram_free_mb: u64, thermal_level: f32 /*0..1, dec_tps_now/baseline*/,
  model_runs_live: u32, cpu_runs_live: u32, worktrees_live: u32, ports_leased: u32,
  dec_tps_ewma: f32
}
// Admission verdict:  Admit = Yes(ResourceGrant) | No(reason) | Defer
```

### A.3 — `WakeReport` (overnight batch result) — §4.7.3

```jsonc
WakeReport {
  batch_id: ULID, ran: "<start> → <end>",
  summary: { goals: u32, succeeded: u32, partial: u32, failed: u32 },
  results: [ { goal: string, status: "done"|"partial"|"failed",
              outcome: string, review_ref: DiffRef?, selection_basis: string?,
              resource: { wallclock_ms, peak_ram_mb, dec_tps_avg } } ],
  needs_review: DiffRef[], thermal_events: u32, total_runs: u32, total_model_seconds: u64
}
```

### A.4 — Remote wire protocol (laptop ↔ server) — §4.9.2

```jsonc
// Transport: persistent WebSocket (wss://, or ws:// over loopback/SSH-tunnel). Framing: JSON-RPC 2.0.
// Auth: bearer token (device-paired) or mTLS; token sent ONLY over wss/loopback. Carries a ch.10 capability grant.

// session/new      → { session: ULID, mcpServers?: [...] }            // ACP-style handshake
// session/resume   → { session: ULID, from_seq: u64, client_token }   // reconnect; server replays (from_seq, head]
// hide/intent      (client→server) { intent: string, session, body }  // ch.01 Wire A over the wire; server acks {accepted, event_seq}
// hide/event       (server→client) { ...ch.01 Event envelope... }     // the projection stream; seq-ordered, exactly-once
// hide/approval    (client→server) { run_id, decision: "grant"|"deny", body? }  // ch.02 Paused/autonomy round-trip

// INVARIANTS:
//  - Server is authoritative; client holds only a rebuildable projection (ch.01 T2).
//  - Reconnect resumes from last durable `seq`: NO duplicate events, NO re-fired effects (ch.01 T3).
//  - Every intent checked against the session's ch.10 capability grant before it appends an event (ch.01 T4).
//  - Sessions persist server-side independent of the connection (a batch survives client sleep/disconnect).
```

### A.5 — New event kinds emitted by this chapter (registered per ch.01 §7.2)

```
job.enqueued {job_id, kind, priority, deps}            — (neither)   — Agent/User
job.admitted {job_id, grant}                           — (neither)   — System
job.started  {job_id, run_id}                          — (neither)   — System
job.preempted {job_id, for, checkpoint_ref}            — Action      — System
job.resumed  {job_id, from_checkpoint}                 — (neither)   — System
job.completed {job_id, status, result_ref}             — Observation — System
workspace.created {run_id, path, ports, base_ref}      — Action      — System
workspace.released {run_id, kept}                       — Observation — System
merge.selected {winner, beaten[], basis}               — (neither)   — Agent
merge.conflict {file, hunks, needs_human?}             — (neither)   — System
merge.resolved {file, by: "ast"|"3way"|"llm"|"human"}  — Observation — System/Agent
merge.completed {adopted[], dropped[], conflicts[]}    — Observation — System
governor.backoff {reason: "ram"|"thermal", new_ceiling}— (neither)   — System
governor.breaker {trigger: "spawn_rate"|..., action}   — (neither)   — System
remote.session_opened {session, client, transport}     — (neither)   — System
remote.reconnected {session, from_seq, replayed_n}     — (neither)   — System
batch.report {batch_id, WakeReport}                    — (neither)   — System
```

---

## Appendix B — Source register

**Orchestration frameworks & patterns**
- LangGraph durable execution / checkpointing — docs.langchain.com, github.com/langchain-ai/langgraph; "checkpoints are not durable execution" (diagrid.io).
- Multi-agent frameworks 2026 (LangGraph/CrewAI/AutoGen/Swarm comparisons) — gurusup.com, presenc.ai.
- OpenAI Swarm → Agents SDK (handoffs/guardrails/tracing; Mar 2025 successor; Mar 2026 Temporal integration) — github.com/openai/swarm, developers.openai.com (orchestration), openai.github.io/openai-agents-python.
- Multi-agent orchestration pattern language (producer/consumer/critic/judge + coordinator; fan-out/pipeline/debate/supervisor/swarm) — digitalapplied.com, lushbinary.com.

**Parallel coding agents & best-of-N**
- SWE-Master post-training; SWE-Replay test-time scaling; DeepSWE (RL, 512-container parallel) — arxiv / together.ai/blog/deepswe.
- SWE-agent batch mode (shared config pool, per-instance isolation) — swe-agent.com/latest/usage/batch_mode.
- "Run k agents, regression-filter, majority-vote/judge survivors" recipe (5–10 / 5 trajectories) — SWE-Master, SWE-Replay.

**Orchestrator-worker (Anthropic)**
- "How we built our multi-agent research system" (lead + 3–5 parallel subagents, own context windows, 90.2% over single-agent, ~15× tokens, token usage = 80% of variance, need crisp objectives) — anthropic.com/engineering/multi-agent-research-system.

**Git-worktree parallel agents & merge**
- Conductor (Melty Labs), Crystal/Nimbalyst, ccswarm, Augment/Upsun worktree guides; VS Code worktree support (Jul 2025).
- Runtime-isolation gap (ports 3000/5432/8080; container-use/Dagger) — penligent.ai, augmentcode.com/tools/open-source-agent-orchestrators.
- Merge: git-merge / merge-strategies docs (recursive 3-way; octopus refuses on conflict); integration-branch funnel + shared task doc — git-scm.com, nrmitchi.com, mindstudio.ai.

**Task queues / DAG scheduling / backpressure / durable execution**
- Distributed job scheduler design (Scheduler/Queue/Workers; Kahn topological ready-set; min-heap priority; weighted fair-share; K8s priority-class preemption) — systemdesignhandbook.com, blog.algomaster.io, geeksforgeeks.org.
- Backpressure in a parallel DAG executor (bounded admission, unbounded completion-reporting) — reymom.xyz.
- Temporal vs LangGraph durable execution (replay event history, resume at exact step, Continue-As-New); idempotency prerequisite — medium/data-science-collective, appscale.blog, diagrid.io.

**Remote agent control**
- Agent Client Protocol (JSON-RPC, local+remote, session/new+session/load, MCP in handshake; Zed Aug 2025, JetBrains Oct 2025) — agentclientprotocol.com, zed.dev/acp, kiro.dev.
- Codex/OpenClaw remote (app-server, SSH-tunneled WebSocket, server keeps working on disconnect, resync on reconnect, wss/loopback tokens) — codex.danielvaughan.com, docs.openclaw.ai.
- AI-streaming durable sessions (server-side persist, reconnect at last-acked offset, no duplicate tokens) — websocket.org, liveblocks.io.

**The sobering limits**
- Single-agent ≥ multi-agent at equal token budget (DPI argument; over-exploration/conformity) — arxiv 2604.02460, beancount.io; "Stop Overvaluing Multi-Agent Debate" arxiv 2502.08788; Multi-LLM-Agents-Debate ICLR-blog 2025.
- MAST taxonomy (spec 41.8% / misalignment 36.9% / verification 21.3%) — Cemri et al. 2025 (NeurIPS).
- Apple-Silicon limits (Metal ~75% RAM cap; concurrent queries share compute; swap collapse; thermal −20–40% after 10–15 min) — stencel.io, macgpu.com (2026 concurrency/queue), solidaitech.com.
- Runaway-agent circuit-breakers (loop detection, EWMA, max-attempts, wall-clock timebox, per-resource ceilings) — fountaincity.tech (cost circuit breaker), blogs.oracle.com (runtime budget guardrails).
