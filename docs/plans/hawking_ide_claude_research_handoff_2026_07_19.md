# Claude handoff: independent Hawking IDE frontier pass

Date prepared: 2026-07-19  
Purpose: give Claude a repository-grounded but independent second research pass before implementation  
Primary input: docs/plans/hawking_ide_frontier_2026_07_19.md

## Copy-paste goal

You are conducting an independent, adversarial architecture and frontier-research pass for Hawking IDE, also called HIDE.

Hawking IDE is the local-first agent platform that runs beside the Hawking inference runtime. It is not merely the Hawking model server and it is not a generic IDE-with-chat proposal. Its job is to turn Hawking's local inference, state, caching, and device economics into the most capable, fastest, highest-performing, and most useful agent-development environment possible.

The owner's priority is lexicographic:

1. capability density;
2. speed;
3. verified performance and quality;
4. usefulness.

Do not silently reorder those goals and do not optimize a proxy such as tokens per second, maximum context length, agent count, benchmark score, or model size in place of the real goal.

Your task is to:

1. internalize the active repository and recoverable HIDE history;
2. independently research the bleeding edge as of the day you run;
3. challenge the existing dossier rather than merely summarize or agree with it;
4. identify capabilities, architectures, research, products, protocols, and implementation levers that the first pass missed;
5. produce a source-linked second-pass synthesis and a reconciled, priority-ordered recommendation;
6. stop before implementation unless the owner separately authorizes implementation.

## Required repository orientation

Start by preserving the current workspace state. Do not modify, delete, or incorporate unrelated user files. At preparation time the branch was codex/hawking-mechanics-thermodynamics at 5d1bf1b9 and the following untracked files belonged to the user:

- tools/condense/mech_fidelity_c.py
- tools/condense/mech_fidelity_d.py

Recheck rather than assuming this is still current.

Read these first:

1. docs/plans/hawking_ide_frontier_2026_07_19.md
2. packs/hawking-hide-desktop.json
3. docs/plans/hide_deep_audit_2026_07_16.md
4. docs/plans/hide_sota_frontier_and_regrade_2026_07_16.md
5. docs/plans/hide_research_menu_2026_06_29.md
6. docs/hide-bible/DESIGN_DOCTRINE.md
7. app/src/ipc.ts
8. app/src/wire.ts
9. app/src/store.ts
10. app/src-tauri/src/main.rs
11. crates/hawking-core/src/engine.rs
12. crates/hawking-core/src/model/rwkv7.rs
13. crates/hawking-serve/src/http.rs
14. the active batching, prefix-cache, stateful, system-prompt, and tool-call code under crates/hawking-serve and crates/hawking-core

The HIDE backend was sealed from the active workspace on 2026-07-17. Audit the latest full source at commit:

    5a99d0e2d7bf7ea822fd41a74f713008bacba1a5

Use git show or a temporary read-only extraction to inspect it. The sealed archive should be available at:

    /Users/scammermike/Downloads/hawking-packs/historical/hawking-hide-desktop/pack.tar.gz

The expected archive SHA-256 is:

    f0c75f9309120f8375256d560bd6670261aec20793d4dde14a85607990ccfa8c

Do not hydrate the pack into the active workspace merely to research it. Do not assume all packed code should return. Determine what deserves reintegration, what should be redesigned, and what should remain archived.

## Repository truths to verify, not merely inherit

The first pass found:

- the React/Tauri frontend is active and still targets hide-serve;
- hide-serve and twelve supporting crates are absent from the active workspace;
- the historical production host bypassed most of its sophisticated kernel, tools, context, and fleet code;
- compiled context was historically discarded;
- rich frontend features were frequently mock, optimistic, or no-op;
- Hawking has useful continuous batching, tool-wire shaping, prefix/state caches, and RWKV state primitives;
- HIDE cannot access state save/load/fork over HTTP;
- direct-admit prefix reuse misses the common batch-one path;
- Serve hardcodes a 4096 sequence limit;
- stop, structured-output, and speculative features are incomplete on the batched path;
- context-window multiplier claims mix weight compression with context and recall;
- local execution security is not yet adequate for an autonomous IDE.

Recheck each claim against current code and history. Report corrections with file and line evidence.

## Research method

### Independence

Treat the first dossier as a hypothesis set and source map, not an authority. Search for:

- evidence published after its cutoff;
- negative results;
- implementation failures;
- incompatible hardware assumptions;
- benchmark contamination;
- security counterexamples;
- simpler baselines that match complex systems;
- capabilities missing from its facet map.

If your conclusion matches the first pass, explain which independent evidence caused the convergence. If it differs, state the contradiction precisely.

### Sources

Use current primary sources wherever possible:

- original papers and proceedings;
- official model cards and technical reports;
- official repositories and specifications;
- official provider/runtime engineering writeups;
- official product documentation for observed product behavior.

Use secondary sources only to discover leads. Do not base a load-bearing recommendation on a search snippet, unsourced post, vendor comparison table, or benchmark leaderboard alone.

For every numerical claim record:

- source and publication date;
- hardware;
- model and precision;
- batch and context regime;
- workload;
- baseline;
- whether it is peer-reviewed, vendor-reported, or an unreviewed preprint;
- whether it has been reproduced in Hawking.

Use these evidence labels:

- VERIFIED REPO
- PRIMARY SOURCE
- INFERENCE
- EXPERIMENT
- CONTRADICTED
- UNKNOWN

### Time and freshness

State your research cutoff in the document. Search the current web; do not assume the July 19 source set remains current. Prefer the newest reliable evidence when a design or product has changed.

## Required research facets

Cover every facet below. Add facets if the decomposition is incomplete.

### 1. Capability density

- open coding-agent model frontier;
- total versus active parameters;
- hybrid recurrent/linear-attention plus sparse-MoE designs;
- FIM, tool use, editing, multimodality, computer use, and agentic training;
- quantization quality by capability;
- model and effort routing;
- whether Qwen3-Coder-Next remains the best first architecture-fit target;
- whether a different local model pool yields a better Pareto frontier.

### 2. Local inference and state

- Apple Silicon Metal and unified-memory runtime frontier;
- batch-one interactive latency versus throughput;
- prefill, decode, expert routing, and memory-bandwidth bottlenecks;
- prompt/radix KV caching;
- KV compression and hierarchy;
- recurrent and hybrid state checkpoint/fork/rollback;
- complete execution-state capsules;
- speculative decoding, multi-token prediction, suffix decoding, and file-as-draft;
- state compatibility and correctness;
- thermals, energy, and contention under multiple agents.

### 3. Effective context

- nominal long context versus reliable retrieval and reasoning;
- exact-code retrieval, AST/symbol/LSP/graph/git/test/trace indexes;
- agentic exploration under a hard budget;
- context compilers and provenance;
- bounded active context plus durable external state;
- compaction continuity;
- memory extraction, supersession, temporal/entity retrieval, and poisoning;
- local/cloud continuity;
- what should be exact source, summary, state, cache, or artifact.

### 4. Tool and agent-loop speed

- stable tool registries and deferred discovery;
- cache-stable prompt architecture;
- programmatic/code-mode tool orchestration;
- safe parallel and asynchronous tools;
- structured/constrained tool-call generation;
- tool-call speculative decoding;
- speculative tool execution and privacy;
- persistent MCP/LSP/DAP/build/test processes;
- output shaping and artifact handles;
- model-to-tool and tool-to-model gaps.

### 5. Agent architecture and fleet

- minimal flat execution loops;
- plan artifacts versus rigid workflow FSMs;
- actor/verifier separation;
- task dependency DAGs;
- worktrees, write leases, and integration agents;
- retry, stall, crash recovery, resume, and cancellation;
- shared build/test daemons;
- large migrations and background work;
- evidence packets and attention management;
- when multi-agent fan-out helps or hurts.

### 6. IDE and product experience

- inline completion and next edit;
- interactive agent editing;
- review, checkpoints, rewind, fork, and takeover;
- Workstation/command-center patterns;
- background and scheduled tasks;
- attention inboxes;
- context transparency and provenance;
- artifact-based review;
- voice, browser, computer-use, debugging, visual verification, and mobile/remote control;
- local/cloud handoff;
- accessibility;
- usefulness measurements such as Keep Rate, accepted edits, rework, interventions, and human time saved.

Study current official behavior from leading products and minimal scaffolds, including at least:

- Claude Code;
- Codex and Symphony;
- Cursor;
- Zed and ACP;
- GitHub Copilot / VS Code;
- Google Antigravity or its current successor;
- Devin/Windsurf;
- Aider;
- mini-SWE-agent;
- any newer system that materially changes the frontier.

Vendor claims are product signals, not neutral benchmark evidence.

### 7. Security, privacy, and trust

- folder trust before project configuration;
- sandbox and VM boundaries;
- filesystem scopes and symlink handling;
- network egress as capability grants;
- credential brokerage and agent identity;
- prompt/tool/repository/browser injection;
- persistent-memory poisoning;
- local MCP versus remote MCP;
- approval fatigue;
- transactional effects;
- auditability without sensitive telemetry;
- autonomous fleet containment;
- privacy-preserving local/cloud routing.

### 8. Protocols and ecosystem

- ACP;
- MCP and MCP Tasks;
- LSP;
- DAP;
- OpenAI-compatible model APIs;
- provider continuation and compaction APIs;
- OpenTelemetry GenAI conventions;
- artifact and checkpoint portability;
- what should be standardized versus Hawking-specific.

### 9. Evaluation

- current status of SWE-bench Verified and Pro;
- Terminal-Bench and infrastructure noise;
- BFCL and tool-use testing;
- ContextBench and retrieval-process evaluation;
- current multilingual and long-horizon coding evaluations;
- private rotating real-work tasks;
- contamination and broken-task audits;
- repeated trials and confidence intervals;
- online metrics and delayed outcome metrics;
- security and recovery testing;
- exact production-harness replay;
- capability density and critical-path measurement.

## Questions you must answer

1. What is the smallest architecture that can make HIDE materially more capable than existing IDE agents?
2. Which existing packed components are high-value assets, and which are complexity traps?
3. What is the honest technical definition of HIDE's “effectively unbounded context”?
4. What state can be reused losslessly, what is lossy, and what is only a cache?
5. Which local model architecture best fits capability density on the target Apple hardware?
6. Which five changes most reduce time-to-verified-change?
7. Which five changes most increase verified success without unacceptable latency?
8. Which features should be deterministic algorithms instead of model calls?
9. Which work should run locally, which in a cloud model, and which in an isolated cloud environment?
10. What is the correct inner agent loop and outer fleet scheduler?
11. What must be true before the product can safely run unattended?
12. Which product surfaces are essential, and which would be decorative complexity?
13. What should HIDE support through ACP/MCP/LSP/DAP rather than invent?
14. Which prior research bets should be killed, deferred, or promoted?
15. What evidence is required before saying fastest, highest-performing, or capability-dense?

## Required adversarial checks

Try to falsify all of these:

- A longer context window improves coding-agent success.
- Recurrent state is a sufficient replacement for exact retrieval.
- Cheap state forks make best-of-N worthwhile.
- More agents reduce wall clock.
- Tool speculation is safe because calls are read-only.
- A high tool-JSON validity rate means tool use is good.
- Qwen3-Coder-Next is the best local Hawking target.
- Weight compression expands usable context.
- KV quantization is quality-neutral on code tasks.
- Prompt caching stays effective under dynamic tools and model routing.
- A stronger model is always slower or more expensive overall.
- Public coding benchmark gains predict real HIDE usefulness.
- Permission prompts are a sufficient safety boundary.
- Restoring all packed HIDE crates is faster than rebuilding the product spine.

## Output contract

Create:

    docs/plans/hawking_ide_claude_frontier_second_pass_<run_date>.md

The second-pass document must contain:

1. executive verdict;
2. research cutoff and methodology;
3. repository corrections with file/line evidence;
4. points of agreement with the first dossier;
5. contradictions and missing facets;
6. a current frontier synthesis for every required facet;
7. a proposed target architecture;
8. a current-versus-target capability inventory;
9. a lexicographic metric system;
10. a priority-ordered implementation ladder with dependencies and exit gates;
11. isolated research bets with kill criteria;
12. security gates;
13. evaluation matrix;
14. owner decisions;
15. a primary-source ledger with direct links.

Also end with a reconciliation table:

| First-pass claim | Claude verdict | Evidence | Action |
|---|---|---|---|

Actions must be one of:

- KEEP
- MODIFY
- DELETE
- EXPERIMENT
- DEFER
- NEEDS OWNER DECISION

Do not overwrite the first dossier. Preserve both passes so disagreements remain legible. You may create a third reconciled document only if the owner explicitly asks.

## Quality bar

The output is not complete if it:

- summarizes the first dossier without independent research;
- treats proposed code as implemented;
- treats packed code as active;
- cites only vendor blogs or only papers;
- repeats a benchmark number without its harness and caveats;
- equates context length with usable memory;
- recommends a feature without a measurable success or kill condition;
- ignores Apple batch-one behavior;
- ignores end-to-end tool latency;
- ignores security containment;
- optimizes public leaderboards instead of real Hawking tasks;
- proposes implementation before preserving workspace state and proving the current baseline.

## Seed sources, not a closed list

Use the first dossier's source ledger as a starting map. Independently revisit at least:

- Qwen3-Coder-Next model card and technical report;
- ContextBench;
- InfiAgent;
- NoLiMa and RULER;
- Anthropic prompt-caching and containment reports;
- OpenAI Codex loop, harness engineering, and Symphony;
- SGLang, vLLM, TensorRT-LLM, and Strata;
- Execution-State Capsules / FlashRT;
- XGrammar-2, ToolSpec, AsyncFC, PASTE, and Ghost Tool Calls;
- MCP and ACP specifications;
- OpenAI's 2026 coding-evaluation audits;
- Anthropic's infrastructure-noise study;
- current official product documentation listed above.

Search for stronger or newer evidence before relying on any seed.

## Final instruction

Be ambitious about the product and conservative about evidence. Hawking IDE can aim to be the most capable and fastest local-first agent environment, but every claimed advantage must map to an implemented path, a measurable critical-path mechanism, a reproducible evaluation, and an honest hardware/model envelope.
