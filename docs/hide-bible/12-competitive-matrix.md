# Chapter 12 — Competitive Matrix

> **Purpose (one line).** Map the coding-agent landscape honestly — who the competitors are, where each is genuinely strong, what HIDE takes from open-source, and why the advantages HIDE holds on the local plane are structurally durable rather than merely ahead-at-the-moment.

**Status:** DESIGN. This chapter is descriptive, not load-bearing. The contracts it references — the event log (Ch.01 §4.5), the tool wire-format (Ch.03 §4.2), the grammar-constrained decode service (Ch.06 §4.5), the personalization flywheel (Ch.06 §4.10), the parallel-agent orchestrator (Ch.09 §4.6), the Seatbelt sandbox (Ch.10 §4.5) — are all specified in the chapters that own them. This chapter synthesizes across them for positioning purposes. As each chapter's implementation matures, the cells in §12.2 will flip from planned to measured; the footnotes carry the current status. **Tier tags used throughout:** **[SHELL-FIRST]** = base app, ships first; **[TIER-2]** = parallel-agent/swarm layer; **[TIER-3]** = workstation/remote mode; **[PLANNED]** = designed now, built later.

---

## Table of contents

1. [Competitive landscape overview](#121-competitive-landscape-overview)
2. [Dimension matrix](#122-dimension-matrix)
3. [Deep dives](#123-deep-dives)
   - [Claude Code](#1231-claude-code)
   - [Cursor](#1232-cursor)
   - [GitHub Copilot Workspace](#1233-github-copilot-workspace)
   - [Cline](#1234-cline)
   - [Aider](#1235-aider)
   - [Continue](#1236-continue)
   - [Void](#1237-void)
   - [OpenHands](#1238-openhands)
   - [OpenCode](#1239-opencode)
   - [Zed Agent](#12310-zed-agent)
   - [Goose](#12311-goose)
4. [The moat analysis](#124-the-moat-analysis)
5. [OSS harvest map (consolidated)](#125-oss-harvest-map-consolidated)

---

## §12.1 — Competitive Landscape Overview

Most tools in the coding-agent category share a structural constraint that is invisible until you examine it: they are thin shells over metered cloud inference. Claude Code, Cursor, GitHub Copilot Workspace, and the cloud configurations of Cline, Continue, and Aider all route every token through a provider's API. The consequence is a cascade of second-order limits that are not bugs but inherent properties of that architecture: every inference costs money, so context must be budget-managed; every session crosses a data boundary, so enterprises with IP or compliance requirements are structurally excluded; the model is a sealed service, so grammar constraints, custom samplers, logit readback, and LoRA hot-swap are simply unavailable; sessions are stateless by design, so the model that helped you last week has no recollection of your codebase; and the rate-limit and quota regime means that running fifty parallel agents overnight is not a product feature but an expense category.

The open-source tools — Cline, Aider, Continue, Void, OpenHands, OpenCode, Goose — acknowledge this by being model-agnostic: they accept any OpenAI-compatible backend, local or cloud. Several (Void, Goose) actively support local models via Ollama or llama.cpp. Being model-agnostic is not the same as being designed around a local model. None of them provide a runtime format purpose-built for Apple Silicon unified memory, a grammar-constrained decode service wired into tool-call emission, a fine-tune-at-condense personalization loop, or a thermal-and-RAM governor that understands it is running on the same chip as the editor. They are front-ends that can *point at* a local backend; they are not co-designed with one.

HIDE is the first coding IDE where the model is the product in the same sense that the IDE is the product. Hawking Industries makes the runtime (`hawking-core`), the HTTP serving layer (`hawking-serve`), the sub-4-bit trellis quantizer (Hawking Condense, producing `.tq` format), and the IDE shell. That vertical integration is what makes the capabilities in §12.4 genuinely durable: a cloud competitor cannot replicate them without ceasing to be a cloud product, and a local-model-agnostic front-end cannot replicate them without building the runtime itself.

### A note on honest framing

This chapter avoids the common competitive-analysis error of comparing each competitor's weaknesses against HIDE's strengths. Every competitor in this list has been used, measured, or studied carefully. The "what they do well" sections are written to be fair; they describe things competitors genuinely do better than HIDE today, or do differently with genuine merit. The "where HIDE exceeds" sections describe structural advantages, not aspirational ones — and they are annotated with tier tags to distinguish what is shipped from what is planned.

The conclusion of this analysis is not "no one else is good." It is: **the competitors are good, and HIDE is the first tool that is good for an entirely different set of structural reasons**. The developers who will use HIDE are not the ones frustrated that Claude Code is slow or that Cursor has bugs; they are the ones who have hit the wall of cloud-only inference — the compliance requirement, the rate limit, the $6,000 bill, the "I want the model to learn my codebase" wall — and found no product that solves it.

---

The landscape as of mid-2026 has three natural tiers:

**Tier A — Cloud-native, frontier-quality.** Claude Code, Cursor (cloud mode), GitHub Copilot Workspace. These are the current quality leaders on raw reasoning. Their structural weakness is the metered cloud dependency: cost, data egress, statelessness, and model opacity are all load-bearing constraints. HIDE's strategy here is not to beat them on pure reasoning quality immediately — that requires a 32B `.tq` model (runtime-testing, not shell-gating) — but to beat them on *reliability* on tool-heavy tasks via constrained decode, and on all the dimensions where local is the only possible answer: cost, privacy, air-gap, persistence, personalization.

**Tier B — Model-agnostic OSS front-ends.** Cline, Aider, Continue, Void, OpenHands, OpenCode, Goose. These are harvested for OSS components (§12.5) and studied for UX patterns. Their structural weakness is the absence of a co-designed runtime: without owning the decoder, every capability that requires decode-level control (constrained generation, sampler tuning, logit readback, KV sharing) is unavailable. HIDE's strategy here is to take what is useful (diff/apply algorithms, repo-map ranking, event schema, MCP client) and build the runtime layer on top that these tools structurally cannot have.

**Tier C — Study-only.** Zed Agent. AGPL-3.0 makes any code incorporation impossible, and Zed's inference is cloud-only. Studied for rendering architecture and plugin sandbox design only.

### State of competition by task type

Not all coding tasks are created equal. The competitive picture varies sharply by task category:

**Tab completion (< 500 tokens in, < 50 tokens out):**
Cursor (cloud) is the leader on quality at ~50ms. HIDE at 7B local is competitive on latency (~80–200ms) and superior on privacy. Grammar-guided prefix completion eliminates syntactic errors. Personalization (Tier-2) flips quality leadership over time.

**Single-session tool-heavy agent task (5–20 tool calls, 1–3 files):**
This is the primary competitive arena for HIDE at shell-first launch. Claude Code is the quality leader; HIDE's grammar-constrained tool calls close the reliability gap significantly. Cline and Aider are the OSS leaders here; HIDE's tiered applier (harvested from both) is strictly more reliable.

**Long-session, multi-file refactor (20–100 tool calls, 10–100 files, across multiple days):**
HIDE's durable event log (Ch.01 §4.5) and living index (Ch.05 §4.9) win structurally. No cloud tool has cross-session memory that is queryable and automatic; HIDE's is both. The agent can resume work from exactly where it stopped, with the full history queryable.

**Parallel agent workstation tasks (overnight, multiple independent subtasks):**
HIDE's unique territory. Claude Code costs ~$375/50-agent-night; Cursor and Copilot Workspace are not designed for this use. HIDE runs it for free. This is the product category that cloud tools will never price-compete on.

**Compliance-required or air-gap codebases (defense, healthcare, finance, legal):**
HIDE is the only viable tool. This is not a competitive comparison — it is a binary gate.

---

## §12.2 — Dimension Matrix

Symbols: ✅ implemented or inherent to the architecture; ⚠️ partial/limited; ❌ absent by design or gap; 🔜 planned (see footnote for milestone); N/A not applicable.

| Dimension | HIDE | Claude Code | Cursor | Copilot WS | Cline | Aider | Continue | Void | OpenHands | OpenCode | Zed | Goose |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Fully local / no cloud required** | ✅[^1] | ❌ | ⚠️[^2] | ❌ | ⚠️[^3] | ⚠️[^3] | ⚠️[^3] | ⚠️[^4] | ⚠️[^3] | ⚠️[^3] | ❌ | ⚠️[^3] |
| **Cost model** | Free forever[^5] | Pay-per-token | Pay-per-token | Subscription | Pay-per-token | Pay-per-token | Pay-per-token | Pay-per-token | Pay-per-token | Pay-per-token | Subscription | Free (self-host) |
| **Model format** | `.tq` (sub-4-bit)[^6] | Proprietary API | Proprietary API | Proprietary API | Any API | Any API | Any API | GGUF/API | Any API | Any API | Proprietary API | Any API |
| **Grammar-constrained tool calls** | ✅[^7] | ❌[^8] | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Custom logit access / samplers** | ✅[^9] | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Persistent durable memory (never reset)** | ✅[^10] | ❌[^11] | ❌ | ❌ | ❌ | ⚠️[^12] | ❌ | ❌ | ❌ | ⚠️[^13] | ❌ | ❌ |
| **Personalization flywheel** | 🔜[^14] | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Parallel agent swarms (free)** | ✅[^15] | ⚠️[^16] | ⚠️[^17] | ⚠️[^18] | ❌ | ❌ | ❌ | ❌ | ⚠️[^19] | ❌ | ❌ | ❌ |
| **KV-cache inter-agent handoff** | 🔜[^20] | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Hardware thermal/RAM governor** | ✅[^21] | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **Offline / air-gap capable** | ✅[^1] | ❌ | ❌ | ❌ | ⚠️[^3] | ⚠️[^3] | ⚠️[^3] | ⚠️[^4] | ⚠️[^3] | ⚠️[^3] | ❌ | ⚠️[^3] |
| **Context stack visibility** | ✅[^22] | ❌ | ⚠️[^23] | ❌ | ⚠️[^24] | ⚠️[^24] | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Sandbox tier** | Seatbelt/microVM[^25] | Seatbelt[^26] | None | None | None | None | None | None | Docker | None | None | None |
| **Open source** | ❌[^27] | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (AGPL) | ✅ |
| **OSS license** | Proprietary | Proprietary | Proprietary | Proprietary | MIT | Apache-2.0 | Apache-2.0 | Apache-2.0 | MIT | MIT | AGPL-3.0 | Apache-2.0 |

### Footnotes

[^1]: After initial model download, HIDE requires no network access. All inference, indexing, memory reads, sandbox execution, and tool calls are local. The `.tq` runtime surfaces a stable localhost HTTP endpoint (Ch.01 §4.3); air-gap is the default, not an opt-in. The only network touch after download is optional: MCP servers that the user explicitly configures to reach external APIs, and the GitHub API bridge for PR-level tasks — both are user-consented, both are off by default.

[^2]: Cursor supports local models via Ollama/OpenAI-compatible backends in its privacy mode, but the core tab-complete and agent mode use Cursor's cloud by default; the UX is designed for cloud. Privacy Mode routes to Cursor's cloud with no-training but is still not local inference.

[^3]: These tools accept a local backend (Ollama, llama.cpp, LM Studio) via an OpenAI-compatible URL but are not designed around one. Air-gap works only when a local backend is manually configured and the tool's telemetry/update checks are disabled. The experience degrades on smaller local models because the tool's prompting strategy was tuned on frontier models.

[^4]: Void's explicit design goal is local-first and supports Ollama natively. Air-gap is more natural here than in other OSS tools, but no `.tq`-format support exists and Void has no thermal governor, grammar-constrained decode, or living index.

[^5]: HIDE's cost model is the hardware the user already owns — an Apple Silicon Mac. Once the model is downloaded, every token, every parallel agent, every retry, every fine-tune run costs $0 in direct spend. Cloud tools cost $5–20/month in subscriptions or $0.003–0.015 per 1K output tokens; at HIDE's typical heavy-user volumes (500K–2M tokens/day across agents), the gap is material. A developer running 20 parallel overnight agents on HIDE pays the same marginal cost as running zero.

[^6]: `.tq` is Hawking Condense's sub-4-bit trellis-quantized format (the absorbed strand-quant STR2 format). A 32B `.tq` model occupies ~18 GB in unified memory versus ~20 GB for Q4_K_M GGUF — the 2 GB difference that determines whether the model fits Apple Silicon's 19 GB effective model budget at all (Ch.06 §4.2; see memory/condense_32b_native_serving_2026_06_23.md). The GPU bitslice GEMV path (`strand_bitslice_gemv_tcb`) is staged; CPU is the parity oracle.

[^7]: The grammar-constrained decode service (Ch.06 §4.5) compiles any JSON-schema or GBNF grammar to valid-next-token masks via the `mask_logits` machinery in `json_constrain.rs`, making tool-call emission valid-by-construction. A small local model that is *constrained* to emit schema-valid tool calls has a format-failure rate of effectively 0, not 15–25% (the measured rate for unconstrained frontier models on schema-shaped outputs, per arXiv:2510.14453). This is available today for `json_mode`; the full schema-driven path is a [SHELL-FIRST] feature. The `JsonConstraint` state machine and `JsonVocabIndex` (token_id→text, built once per model) already exist in `crates/hawking-core/src/json_constrain.rs`.

[^8]: Anthropic's strict tool use (`strict:true`, GA Nov 2025) does apply schema-enforced sampling server-side, which is the same mechanism. The distinction is that HIDE's user can supply *any* grammar for *any* local model, not just the schemas Anthropic chooses to enforce on their servers. HIDE's grammar service is a user-programmable capability, not a server-side policy.

[^9]: Because HIDE owns the decoder, Ch.06 §4.6 specifies sampler profiles (deterministic for edits, exploratory for brainstorming, low-entropy for constrained generation) that extend the current `SamplingParams { temperature, top_k, top_p, repetition_penalty, seed }` with min-p, typical-p, DRY, and logit-bias. Logit-level features — token confidence, entropy gating, self-certainty, self-consistency voting — are exposed as agent-callable signals (Ch.06 §4.7). The Metal GPU path keeps logits on-GPU; readback is a deliberate, gated feature (the bus crossing is not free).

[^10]: The append-only event log (Ch.01 §4.5) is hash-chained and never truncated. Every session is a contiguous segment of the same log; "starting a new session" means advancing a read pointer, not deleting history. The living-index daemon (Ch.05 §4.9) runs continuously, not session-scoped. Memory is not a context window — it is a queryable store (Ch.04 §4.x) that can be retrieved by semantic similarity, exact symbol name, session ID, or time range, across all past sessions without any curation by the user.

[^11]: Claude Code's CLAUDE.md and project memory files are a workaround, not a durable session store. Each CLI invocation starts fresh; the "memory" is static text files the user curates and must remember to update. There is no automatic capture of session state, no query API over past sessions, and no daemon that keeps context fresh.

[^12]: Aider maintains a `.aider.chat.history.md` log and a `--chat-history-file` that it can re-read, but this is a flat text dump, not a queryable event log. Session continuity is partial and manually managed.

[^13]: OpenCode has session management with JSON-backed session files and can resume sessions by ID. This is closer to HIDE's model than most CLI tools but lacks the hash-chained durability, replay semantics, and daemon-maintained living index.

[^14]: The fine-tune-at-condense flywheel (Ch.06 §4.10) captures accepted diffs → teacher-forced KD dataset → Condense trainer → updated local checkpoint. This is a [TIER-2] feature (POST-SHELL). The data-capture seam (recording accepted diffs in the event log) is [SHELL-FIRST]; the trainer invocation is a Condense-side concern (tools/training/).

[^15]: The parallel-agent orchestrator (Ch.09) runs dozens of Ch.02 agent loops simultaneously, worktree-isolated, with a resource governor respecting Apple Silicon's RAM and thermal budget. Each run costs $0 in direct spend. The orchestrator ships as [TIER-2] (SWARM), designed fully in Ch.09 so no further design is needed when the tier is built.

[^16]: Claude Code's subagent spawning (the `--dangerously-skip-permissions` flag and the `Task` tool for calling sub-agents) is supported but each spawned agent is a separate API call billed per token. Running 50 parallel agents is a budget decision, not a free architectural property.

[^17]: Cursor's background agent mode allows running agents on tasks while the editor is open but is backed by cloud inference; parallel agents multiply cost linearly.

[^18]: GitHub Copilot Workspace supports parallel tasks across PRs but each task hits Azure-hosted inference on GitHub's infrastructure.

[^19]: OpenHands's multi-agent framework supports spawning child agents (BrowsingAgent, ReadOnlyAgent, etc.) but each agent call hits a cloud API in the default configuration.

[^20]: KV-cache prefix sharing across agents (Ch.06 §4.11, Ch.09) allows N worktree agents sharing a system prompt or repo context to share the KV prefix — only one model forward pass for the common prefix, reducing re-prefill cost by `O(prefix_tokens)` per agent launch. This is a runtime-side feature requiring `copy_kv_prefix_to_slot` (already in the `Engine` trait) wired into the orchestrator. [TIER-2 / PLANNED].

[^21]: The resource governor (Ch.09 §4.6) reads Apple Silicon's IOReport thermal metrics and NVRAM RAM budget, adjusts the number of active model slots and batch sizes, and preempts lower-priority batch agents when an interactive request arrives — losslessly, via the Ch.02 checkpoint/resume mechanism (Ch.09 §4.6 preemption). No cloud tool has a hardware governor because cloud tools don't share the hardware with the user.

[^22]: The Context Stack panel (Ch.07) shows the user exactly what is in the agent's context window: the assembled spans, their provenance tags (trusted/untrusted per Ch.10 §4.7), their token cost, and which retrieval path produced them (Ch.04). The user can pin, exclude, or rerank items before dispatch. This is the inverse of the opacity that characterizes cloud agents, where the prompt is assembled inside the provider's infrastructure and never shown.

[^23]: Cursor's context pill UI shows which files are referenced but does not show the assembled prompt, the exact spans, or the retrieval path. The user sees references, not content.

[^24]: Cline's and Aider's verbose logging modes (`--verbose` in Aider, Cline's debug panel) show tool calls and responses but not the assembled context window prior to model dispatch, and not the token cost breakdown.

[^25]: Ch.10 §4.5 specifies a tiered sandbox: Tier 0 (worktree confinement + capability-scoped tools), Tier 1 (macOS Seatbelt `sandbox-exec` profile with network-deny), Tier 2 (Apple `container` microVM or gVisor). Network is default-deny via a host-side allowlist proxy (Ch.10 §4.5.4). This is structurally comparable to Anthropic's own `sandbox-runtime` for computer-use and is stricter than Docker in that the network proxy applies a domain allowlist rather than just isolating the container.

[^26]: Claude Code (in its computer-use and shell execution modes) runs subprocesses in macOS Seatbelt on macOS, but only for the tool execution subprocess — not for the broader session state.

[^27]: HIDE is proprietary. The Hawking runtime crates (`hawking-core`, `hawking-serve`) are not published under an OSS license. The OSS harvest (§12.5) incorporates permissively licensed components into HIDE's codebase under the terms of their respective licenses; those terms are documented per-component in §12.5. Zed's AGPL-3.0 code is studied but never copied (§12.3.10).

### §12.2.1 — Reading the matrix

The matrix encodes three kinds of difference:

**Structural differences** — things that cannot be added to a tool without rebuilding it from the ground up.

- "Grammar-constrained tool calls" requires owning the token-level sampler. A tool that calls an external API cannot add this no matter how much engineering it invests; the API provider would have to add it, and even then only for their own models.
- "Personalization flywheel" requires owning the fine-tuning pipeline and the model weights. A tool that calls Claude or GPT-4 cannot build a personalized version of those models; Anthropic and OpenAI will not provide a per-user fine-tuning API for proprietary model weights.
- "Hardware thermal/RAM governor" requires running on the same hardware as the model. A cloud tool governs its own datacenters, not the user's laptop; this dimension is simply N/A for every cloud tool.

**Engineering differences** — things that are technically possible for a competitor to add but represent significant, months-long work.

- "Persistent durable memory (never reset)" could be added to Claude Code if Anthropic invested in a server-side session store with an append-only log and a query API. It would take months and would still not solve the privacy concern (data is on Anthropic's servers).
- "Parallel agent swarms (free)" could be added to any cloud tool if the operator accepted the cost. At $0.015/1K tokens, "free" is the wrong word; the correct version is "unlimited swarms at $X/swarm."
- "Context stack visibility" is buildable for any tool; it is primarily a product-investment question. Cursor has moved in this direction with context pills; full span-level visibility is further work.

**Deliberate absences** — things competitors have chosen not to build.

- "Open source" for HIDE is a deliberate choice: the runtime and the quantizer are the moat. Publishing them as OSS would gift the stack to competitors.
- "Sandbox tier" for Cursor, Aider, Continue, OpenCode, Goose is a deliberate non-investment: these tools are designed as front-ends to a remote API where the blast radius of a bad command is the user's local filesystem, and the user has accepted that risk.

Understanding these three categories prevents false comparisons: saying "Claude Code could just add local inference" is like saying "a streaming service could just add terrestrial broadcast." Technically true, categorically transformative.

---

## §12.3 — Deep Dives

### Competitor strengths summary

Before diving into individual analyses, this table captures each competitor's genuine primary strength — the one dimension where they are the best choice.

| Competitor | Primary strength | The one thing they do better than anyone |
|---|---|---|
| **Claude Code** | Raw reasoning quality | Sonnet 4 / Opus 4 on hard multi-file refactors and architectural planning |
| **Cursor** | IDE integration + tab-complete speed | Zero-migration VSCode fork; cloud fast-apply at ~1000 tok/s |
| **Copilot Workspace** | GitHub-native issue→PR pipeline | Native GitHub context without copy-paste |
| **Cline** | Most actively developed OSS agent | Comprehensive tool loop + battle-tested diff/apply |
| **Aider** | Repository awareness | Best-in-class repo-map via ctags+PageRank |
| **Continue** | Model-agnosticism + JetBrains | Works with any backend; JetBrains is unique in OSS |
| **Void** | Local-model-first OSS IDE | Best starting point for a local-model VSCode fork |
| **OpenHands** | SWE-bench score (OSS) | Best published OSS agent on complex multi-step tasks |
| **OpenCode** | Session management (CLI) | Cleanest session-resume UX in a terminal agent |
| **Zed** | Editor responsiveness | Sub-millisecond keypress latency via GPUI |
| **Goose** | MCP client quality (Rust) | Best production-quality Rust MCP client (`rmcp`) |

HIDE's primary strength at shell-first launch: **zero marginal cost + air-gap + grammar-constrained decode + durable session memory.** The quality/reasoning leadership comes with the 32B `.tq` tier and the personalization flywheel.

---

### §12.3.1 — Claude Code

**What they do well.** Claude Code is the most capable coding agent in production today on hard multi-file reasoning tasks. The Sonnet 3.7/4 and Opus 4 models excel at decomposing large refactors, understanding deep call graphs, and generating coherent multi-step plans with minimal hallucination on well-specified problems. The CLAUDE.md memory convention, while manual, is practical and understood by power users. The permission model is honest and explicit: the tool presents each destructive action before executing. The `--dangerously-skip-permissions` flag and the `Task` subagent tool make multi-step automation usable in batch contexts. The tool loop — `read_file`, `edit`, `bash`, `grep`, `glob` — is well-tested at scale across a broad range of real codebases, and the edit-apply quality on Sonnet 4+ is high. The CLI interface is composable with Unix tooling in ways that a GUI IDE is not. The model's in-context understanding of entire repo structures at 200K+ token context windows is a genuine strength that 7B local models cannot match today.

**Where HIDE exceeds them.** Claude Code has no local inference path: every token is billed, every session crosses Anthropic's infrastructure, and no data stays on the machine. For an enterprise developer working on a proprietary codebase, this is a structural blocker, not a preference — it is a compliance problem that cannot be waivered away. HIDE's event log gives continuity across sessions that CLAUDE.md cannot approximate: the log is queryable, durable, hash-chained, and daemon-maintained (Ch.01 §4.5, Ch.04), not a static text file the user must curate. Grammar-constrained tool calls (Ch.06 §4.5) mean a HIDE 7B local model emits valid tool-call JSON with a format-failure rate of ~0%, while unconstrained frontier models at Anthropic's scale still see 15–25% semantic errors inside valid-schema outputs (arXiv:2510.14453). Parallel agent swarms cost HIDE $0; running 50 Claude Code subagents overnight is a measurable API bill — at $0.015/1K output tokens and 500K tokens per agent session, that is $375 per overnight run. The personalization flywheel (Ch.06 §4.10) means HIDE's model learns your codebase's idioms on accepted diffs; Claude Code's models are stateless per-session. The Context Stack panel (Ch.07) shows the user exactly what the agent sees; Claude Code's context assembly is opaque.

**What HIDE harvests.** Nothing directly — Claude Code is proprietary with no published source. Its tool-loop design (the `read_file` / `edit` / `bash` / `grep` / `glob` core set), the CLAUDE.md convention as a user-editable project context, and the permission-display-before-execute UX are all studied for the Ch.03 tool catalog and Ch.07 UX design. No code is ported.

**Gap we close at which milestone.** On raw reasoning quality for the hardest tasks — adversarial refactors, subtle bug hypotheses, architecture-level planning over 100+ file codebases — frontier cloud models will remain ahead of any local 7–32B model for a significant period. HIDE closes the quality gap on well-scoped, tool-calling-heavy tasks via grammar constraints (Ch.06 §4.5), logit-guided confidence gating (Ch.06 §4.7), and oracle-first verification (Ch.02). [SHELL-FIRST] target: on tool-heavy tasks where the model loop involves 5–20 tool calls (read, edit, test, grep), a constrained 32B `.tq` model should match or exceed the *reliability* — not the raw reasoning quality, but the task-completion rate — of an unconstrained frontier model on concrete, checkable coding tasks. The reasoning quality gap narrows as `.tq` 32B serving matures ([RUNTIME-TESTING], not shell-gating).

---

### §12.3.2 — Cursor

**What they do well.** Cursor is the dominant commercial AI IDE by mindshare in 2025–2026. Its VSCode fork means zero migration cost for the majority of developers — every VS Code extension, keybinding, and workspace setting carries over. The tab-complete model (a proprietary fine-tuned model, ~GPT-4o-class, served from Cursor's infrastructure) is genuinely faster than a standard API call: Cursor's "instant apply" uses speculative-decoding full-file rewrite at ~1000 tokens/second (confirmed by the Fireworks writeup of their fast-apply model and the Cursor blog), faster than any single-stream model inference. The agent mode with `yolo` (fully autonomous) / supervised step is polished. The `@codebase` semantic search retrieves relevant context across the full repo. The background agent (PR-level task execution, running in the cloud on Cursor's infrastructure) is a real product that ships PRs without the user waiting at the keyboard. The model picker — Sonnet, o3-mini, GPT-4o, Gemini — gives users choice within a coherent UX.

**Where HIDE exceeds them.** Cursor's local-model support is surface-deep: Cursor Privacy Mode routes to Cursor's cloud with a no-training guarantee, not to a local model. The Ollama integration (`Settings > Models > +Add model`) is functional but unofficial — Cursor's tab-complete, context assembly, and agent-mode prompting strategies were not designed for 7B models and degrade accordingly. HIDE's model runs entirely on the user's hardware from the first token. The moat Cursor cannot close structurally: zero marginal cost means HIDE can run "instant apply"-style speculative rewrite *and* 50 parallel agents *and* full-repo embedding updates simultaneously, all for free, all air-gapped, all night. Cursor's tab-complete quality depends on Cursor's fleet RLHF; HIDE's fine-tune-at-condense flywheel (Ch.06 §4.10) trains on the *user's own* accepted diffs — a strictly better signal for that specific codebase. The grammar-constrained tool call path (Ch.06 §4.5) gives HIDE a format-failure rate that Cursor's cloud backends cannot guarantee because they do not constrain tool-call JSON at the token level for all models. Cursor has no durable memory, no living index daemon, no thermal governor, and no personalization flywheel.

**What HIDE harvests.** Nothing from Cursor (proprietary). The "instant apply" speculative-rewrite concept — using the original file as the draft for a full-file rewriter — is studied; HIDE's tiered applier (Ch.03 §4.7) achieves edit reliability via exact-match → fuzzy → AST-aware rather than a trained full-file rewriter, though the full-file rewrite tier is noted as a [MOONSHOT] in Ch.11.

**Gap we close at which milestone.** Cursor's tab-complete latency on an M3 Mac with cloud backend is ~50–150ms. HIDE's local completion at 7B is ~80–200ms (measured on Q4_K_M serve; `.tq` GPU path narrows this). The latency gap is within the same UX tier. The quality gap on next-token completion will persist until HIDE's personalization flywheel has accumulated enough training signal — probably 4–6 weeks of daily use. [SHELL-FIRST] milestone: ship completion with a grammar-guided FIM prefix that forces syntactically valid continuations even at 7B, eliminating the category of "completion finishes in the wrong syntactic position" failures.

---

### §12.3.3 — GitHub Copilot Workspace

**What they do well.** Copilot Workspace occupies the task-to-PR tier: given a GitHub issue, it proposes a plan, assigns changes across files, and opens a PR with a detailed summary. The GitHub integration is native — it reads issue context, PR comments, CI results, linked issues, and repository metadata without the user copying and pasting. The iterative plan editor (human reads and edits the proposed step list before the agent executes) is the correct UX for autonomous large-scope tasks: it gives the developer a legible checkpoint before any edits are made. The enterprise SSO/audit integration via GitHub Advanced Security is necessary for regulated industries and is mature. The scale of GitHub's training data advantage (100M+ repositories) feeds Copilot's in-context understanding of third-party library APIs.

**Where HIDE exceeds them.** Copilot Workspace is a pipeline, not a local agent: the execution happens in GitHub's Azure-hosted infrastructure with no access to the local machine's state — its build system, its proprietary internal libraries, its running services, its CI secrets. HIDE's parallel-agent orchestrator (Ch.09) can run the same task-to-PR flow on worktree-isolated agents with real local build and test oracles (Ch.02 §4.6). The agent *knows* whether the code compiles and whether the tests pass before opening a PR, not after; the oracle-gated merge (Ch.02) means HIDE will not propose a PR whose tests fail. Copilot Workspace's CI gate is post-PR-open and asynchronous — the PR is already on GitHub before the failure is known. Air-gap and no-data-exfiltration are, again, structural impossibilities for a pipeline that runs on GitHub infrastructure. HIDE's durable event log (Ch.01 §4.5) provides a tamper-evident audit trail; Copilot Workspace's server-side logs are under Microsoft/GitHub's control.

**What HIDE harvests.** Nothing (Microsoft, proprietary). The plan-then-implement UX pattern is studied. The idea of a structured issue→plan→edit→PR flow informs HIDE's Ch.09 batch-job schema (Ch.09 §4.5).

**Gap we close at which milestone.** GitHub-native context (issues, CI status, PR review comments, code review history) requires an authenticated API call. HIDE surfaces this via an MCP bridge to the GitHub REST/GraphQL API (Ch.03 §4.8) so agents can read issue context and push PRs. [SHELL-FIRST] feature: the GitHub MCP server is part of the default tool catalog. The full task-to-PR pipeline with oracle-gated merge is [TIER-2].

---

### §12.3.4 — Cline

**What they do well.** Cline (MIT, VSCode extension) is the most actively developed OSS agentic coding tool as of mid-2026. Its tool loop is comprehensive: `read_file`, `write_file`, `execute_command`, `search_files`, `list_files`, `browser_action`, `list_code_definition_names`, `attempt_completion`. The diff/apply engine is battle-tested: Cline migrated from XML tag format to native JSON tool calls in v3.35 (citing ~100% multi-tool reliability on frontier models), and the `replace_in_file` applier uses `------- SEARCH` / `=======` / `+++++++ REPLACE` blocks with model-tuned fallback matching logic. The plan mode (model proposes a plan, human approves before execution) is a UX that HIDE adopts. The MCP client integration is production-quality and covers both stdio and HTTP transports. Cline's auto-approve configuration (per-tool, per-domain) gives power users fine-grained control over what requires confirmation. The project has a large contributor base and ships updates frequently.

**Where HIDE exceeds them.** Cline is a VSCode extension with no local inference stack and no model format: it is a front-end that points at any API. Its diff/apply brittleness on whitespace and ambiguity is a known issue that Roo-Code (a Cline fork) measured at >15% failure on `apply_diff` for certain prompt styles; Cline's own migration from XML to JSON was motivated by format reliability concerns. HIDE's tiered applier (Ch.03 §4.7) — exact-match → whitespace-normalized → fuzzy → AST-aware (Ch.05) — is designed explicitly to eliminate this failure mode at every tier, with a `CONFLICT` response that includes a `fix_hint` on miss rather than silently applying a wrong edit. Grammar-constrained decode means HIDE never needs the `------- SEARCH` marker parsing path at all in the default local flow; the model emits a structured `edits[]` array by construction (TT2 in Ch.03 terms). Cline has no durable memory across sessions, no living index daemon, no thermal governor, and no parallel-agent layer. Every tool call crosses a billing boundary; HIDE's do not.

**What HIDE harvests.** Diff/apply engine (port, MIT): the `replace_in_file` matching algorithm — exact string match → whitespace-normalized match → fuzzy match with similarity floor → `CONFLICT` with candidate hint — is the logic HIDE adopts as Tier-1 of Ch.03 §4.7's tiered applier. HIDE also normalizes Cline-compatible `------- SEARCH` / `+++++++ REPLACE` fences (alongside Aider's `<<<<<<< SEARCH` / `>>>>>>> REPLACE`) as alternate surface formats for the structured edit representation — supporting these fences means HIDE accepts tool-call output from any model or tool that targets either format, without requiring re-prompting. HIDE crate/module: `hide-tools/src/edit/apply.rs`. License compliance: MIT, attribution in NOTICE file.

**Gap we close at which milestone.** [SHELL-FIRST]: the tiered applier and Cline-compatible fence normalization are shell-first features. The OSS harvest is a port of the matching logic, not a dependency on Cline's VS Code extension infrastructure.

---

### §12.3.5 — Aider

**What they do well.** Aider (Apache-2.0, CLI) has the most sophisticated repository-awareness of any OSS coding agent. Its repo-map algorithm — tree-sitter parsing of the full repo → ctags-style symbol/reference extraction → PageRank over the symbol dependency graph → token-budgeted selection of the highest-ranked symbols — produces a compact, maximally-informative map of the codebase for any query. This map runs at startup and on each request, not continuously, but even a cold-start map is more precise than embedding-only retrieval on large codebases with complex dependency graphs. The architect/editor mode (one frontier model plans, one cheaper model applies) is an effective cost/quality tradeoff. The auto-model leaderboard (tested edit formats per model, updated continuously) is a genuine public-goods contribution. Aider's git integration (auto-commit on accepted edits, `--dry-run`, `--no-auto-commits`, structured commit messages) is polished. The `--subtree-only` and `--read` flags for scoping context are practical for monorepos.

**Where HIDE exceeds them.** Aider is a CLI tool with no editor surface, no daemon, no living index, no durable memory across sessions, and no sandbox. Its repo-map is computed on-demand at invocation (not maintained continuously), and its scope is symbols extracted by ctags patterns — it does not maintain a call graph, import graph, type graph, or test-coverage graph. HIDE's Living Index (Ch.05) is a standing daemon that keeps the full symbol/reference/call graph always-fresh by watching for file changes (Ch.05 §4.9), so retrieval latency is index-lookup (~milliseconds), not re-parsing (~seconds for large repos). The repo-map algorithm's PageRank signal is one of several signals Ch.05 uses (Ch.05 §4.6); Ch.05 adds call-graph edges (reverse-callee lookup), test-coverage mapping (which tests cover symbol X), import-graph weight (which modules are most imported), and embedding-based semantic similarity — a richer ranking than Aider's ctags+PageRank. Grammar-constrained decode, durable memory, parallel agents, and the Seatbelt sandbox are all absent from Aider by design.

**What HIDE harvests.** Repo-map algorithm (port → extended, Apache-2.0): the tree-sitter parse → symbol/reference extraction → PageRank ranking → token-budgeted selection algorithm is ported and extended as Ch.05 §4.6's ranking pass. HIDE's version adds call-graph edges, test-coverage signal, edit-frequency weighting (recently touched files rank higher), and a hybrid re-rank step that combines structural PageRank with embedding cosine similarity. The ported algorithm is the starting point; the extensions are original. HIDE crate/module: `hide-index/src/repo_map.rs`. License compliance: Apache-2.0, attribution in NOTICE file.

**Gap we close at which milestone.** [SHELL-FIRST]: the repo-map ranking is a shell-first feature (Ch.05 §4.6 is load-bearing for the Context Compiler in Ch.04 — it is the structural retrieval leg that embedding-only search cannot provide). The full call/import/type graph expansion beyond Aider's ctags scope is a [TIER-2] enhancement once the daemon is running continuously.

---

### §12.3.6 — Continue

**What they do well.** Continue (Apache-2.0, VSCode/JetBrains extension) is the most model-agnostic OSS tool: it supports dozens of providers (Ollama, LM Studio, OpenAI, Anthropic, Mistral, Gemini, Together, Replicate, local llamafile) with a single provider abstraction. Its context providers — `@codebase`, `@file`, `@docs`, `@web`, `@terminal`, `@diff`, `@repo-map` — give the user explicit per-message control over what enters the context. The JetBrains support is unique among OSS tools and reaches Java/Kotlin/IntelliJ developers that other tools miss. Tab-complete with fill-in-the-middle (FIM) works with local models including Ollama. The retrieval glue — chunking, embeddings via a local embedding model, lexical BM25 + rerank pipeline — is the most reusable OSS retrieval component in the landscape. Continue's `.continuerc.json` configuration is expressive and supports custom slash commands, custom context providers, and per-workspace settings.

**Where HIDE exceeds them.** Continue is a retrieval front-end: it does not own the index, the daemon, the inference stack, or the model format. Its `@codebase` retrieval is session-scoped — re-indexed on first query per session, not maintained continuously by a daemon. HIDE's Living Index (Ch.05 §4.9) runs continuously, maintains the full symbol graph and embedding index always-fresh by file-watching, so retrieval latency is a cache read, not a re-parse. Continue has no grammar-constrained decode (it cannot enforce tool-call schemas at the token level), no durable memory (context providers are query-time, not persistent), no sandbox (shell execution is unbounded), and no parallel agents. Its model-agnosticism means it cannot exploit `.tq`-format-specific features: grammar masks, logit readback, KV sharing, sampler profiles.

**What HIDE harvests.** Retrieval glue (study → port, Apache-2.0): Continue's chunking strategy (function-boundary chunking with sliding-window overlap, language-aware split points via tree-sitter), the FIM-template construction for tab-complete context (prefix/suffix/filename triple), and the lexical-first → embedding → re-rank query pipeline (BM25 + cosine similarity + cross-encoder re-rank) are studied and the relevant algorithmic logic is ported into `hide-index/src/retrieval.rs` (Ch.05 §4.7). Continue's provider abstraction is studied as a reference for HIDE's model-provider plugin design (Ch.01 §4.3). HIDE crate/module: `hide-index/src/retrieval.rs`. License compliance: Apache-2.0.

**Gap we close at which milestone.** [SHELL-FIRST]: the hybrid retriever (Ch.05 §4.7) is a shell-first feature. Continue's FIM template logic informs HIDE's completion-request construction in the tab-complete path.

---

### §12.3.7 — Void

**What they do well.** Void (Apache-2.0, VSCode fork) is the open-source Cursor clone built explicitly for local-model-first use. Its explicit design goal is to be "Cursor but open-source and local," which is a closer starting point for HIDE's UX layer than any other OSS tool. It supports Ollama, LM Studio, and custom OpenAI-compatible endpoints natively as first-class configuration, not a workaround. The Monaco diff UX — side-by-side diff, inline ghost text, accept/reject at the hunk level — is taken directly from VS Code's diff editor and is the reference UX for proposing edits before applying them. Void's dock layout (collapsible side panels, chat panel, file tree, configurable split views) is well-designed for an IDE that needs an agent conversation panel alongside the editor. Void's positioning as an Electron/Monaco-based local IDE makes its UI code directly applicable to HIDE's editor surface.

**Where HIDE exceeds them.** Void is a VSCode fork and inherits VS Code's Electron/Node.js architecture — which means it cannot use WASM Component sandboxing for plugins (VS Code extensions run in full Node.js), does not have a Tauri-native IPC bus with the channel-based ordered streaming semantics (Ch.01 §4.4), cannot call directly into a Rust runtime sidecar without JSON serialization overhead, and has no mechanism for grammar-constrained decode or logit access at all. Void's local model support is identical to Continue's: it points at an OpenAI-compatible URL. HIDE's Tauri 2 host owns the WebView, the IPC bus, and the model sidecar in the same process tree — the round-trip latency for a tool-call response from the model to the UI, the capability depth, and the security posture are categorically different. Void has no durable memory, no living index daemon, no thermal governor, no parallel-agent layer, and its plugin model is VS Code extensions (full Node.js, no sandbox).

**What HIDE harvests.** Monaco diff UX (study → port, Apache-2.0): Void's diff-accept/reject-at-hunk-level UX implementation — the `DiffEditor` React component wrapper, hunk-level accept/reject controls rendered as inline actions, ghost-text rendering for streaming suggestions — is studied and adapted for HIDE's editor surface (Ch.07). The dock/panel layout implementation (collapsible panels with stored widths, split-view configuration) is adapted for HIDE's panel system (Ch.07 §4.x). HIDE crate/module: `hide-ui/src/diff_view.rs`, `hide-ui/src/layout.rs`. License compliance: Apache-2.0.

**Gap we close at which milestone.** [SHELL-FIRST]: the diff view and dock layout are shell-first UI features (Ch.07). The Monaco editor integration is core to the IDE shell. Note that HIDE uses Tauri 2's WebView to host Monaco/CodeMirror rather than Electron, which gives a smaller binary size and native OS integration, but the editor component itself (Monaco) is the same.

---

### §12.3.8 — OpenHands

**What they do well.** OpenHands (MIT, formerly OpenDevin) has the most rigorous agent-runtime design of any OSS tool. The `CodeActAgent` encodes actions as typed Pydantic objects — `CmdRunAction`, `IPythonRunCellAction`, `FileEditAction`, `BrowseInteractiveAction` — and results as typed `Observation`s (`CmdOutputObservation`, `FileReadObservation`, etc.), with a strict `cause` field linking each Observation to the Action that produced it. The event-stream architecture — append-only, replayable, JSON-serialized — is the closest OSS analogue to HIDE's event log design. The `Runtime` Docker container provides genuine process isolation: each session gets a fresh container with a controlled filesystem and network environment. The multi-agent framework (specialist subagents — BrowsingAgent, ReadOnlyAgent — spawned from a parent orchestrator) is a production implementation of the fan-out pattern described in Ch.09. The SWE-bench scores are the best published for any OSS agentic system as of mid-2026.

**Where HIDE exceeds them.** OpenHands runs in Docker: it requires a running Docker daemon, is not native on macOS without Rosetta/Colima overhead (the daemon startup adds seconds to every session start), and the per-session container model means cold-start latency for the sandbox is ~2–5 seconds. HIDE's Seatbelt sandbox (Ch.10 §4.5) provides equivalent process isolation natively on macOS, with ~100ms warm-up via `sandbox-exec` profile application. OpenHands's Docker-per-session model does not share KV cache across sessions; HIDE's KV-prefix sharing (Ch.06 §4.11) reduces re-prefill cost when multiple agents share a common system prompt prefix. OpenHands has no grammar-constrained decode, no durable memory across sessions (beyond the event stream export), no living index daemon, no thermal governor, and its local-model support is again the OpenAI-compatible URL approach.

**What HIDE harvests.** Timeline and event model (study → port, MIT): the `EventStream` architecture — append-only events, `Action`/`Observation` typed union, `cause`-link from observation to action, `source` tagging (agent/user/environment/tool) — is the direct ancestor of HIDE's Ch.01 §4.5 event schema. HIDE's `tool.call` / `tool.result` event pair with `cause` linking is a direct port of OpenHands's action/observation cause-link. The `source` tagging (which part of the system emitted the event — agent loop, user, tool execution, environment) maps to HIDE's Ch.01 event schema's `actor` field. HIDE crate/module: `hide-core/src/event.rs` (the schema; Ch.01 owns it). The `CodeActAgent`'s CodeAct design (Python-execution-as-tool) informs HIDE's Ch.03 §4.6 `code.exec` tool entry. License compliance: MIT.

**Gap we close at which milestone.** [SHELL-FIRST]: the event log (Ch.01 §4.5) and the capability-scoped tool sandbox (Ch.03 §4.9, Ch.10 §4.5) are shell-first features. Parallel multi-agent with Docker replaced by worktree isolation is [TIER-2] (Ch.09). The SWE-bench score gap is a reasoning-quality gap that the 32B `.tq` model targets; it is [RUNTIME-TESTING].

---

### §12.3.9 — OpenCode

**What they do well.** OpenCode (MIT, TUI) has the most polished session management of any OSS coding CLI. The session list, session resume by ID, and session export UX are clean and keyboard-navigable. The plan/act mode — the agent proposes a numbered step list and executes step-by-step with user confirmation between steps — is the right UX for interactive autonomous tasks: the user sees the full plan, not just the current step. OpenCode's TUI is responsive in the terminal and requires no Electron/browser overhead. The tool set (read, write, execute, search) is comprehensive for a CLI context. OpenCode reads and respects AGENTS.md for per-session context, similar to Claude Code's CLAUDE.md but scoped to the OpenCode session.

**Where HIDE exceeds them.** OpenCode is a TUI with no editor surface, no living index, no durable memory (session files are flat JSON on disk, not a hash-chained log), no sandbox (shell execution is unbounded), no grammar-constrained decode, and no local model format support beyond an OpenAI-compatible URL. Its session management, while good for a CLI, is not queryable as a structured store — you cannot ask "what did the agent do to module X two sessions ago." HIDE's event log (Ch.01 §4.5) is queryable, replayable, and tamper-evident. The plan/act UX pattern is good but the implementation is text-only; HIDE's equivalent (Ch.07) is rendered in a structured panel with inline diff previews for each step before approval.

**What HIDE harvests.** Session UX (study, MIT): OpenCode's session-list → session-resume → step-through-plan UX pattern is studied for HIDE's session browser (Ch.07). The plan/act interaction model (numbered steps, step-level confirmation) informs HIDE's supervised-step autonomy level (Ch.02 §4.3). No code is ported — the TUI implementation is not applicable to HIDE's WebView-based surface.

**Gap we close at which milestone.** [SHELL-FIRST]: the session browser and plan/act UX are shell-first features (Ch.07).

---

### §12.3.10 — Zed Agent

**What they do well.** Zed (AGPL-3.0, native Rust editor) is technically the most sophisticated editor in the competitive set. Its GPUI framework (a custom retained-mode GPU rendering library in Rust) achieves sub-millisecond keypress-to-screen latency on Apple Silicon, a level of editor responsiveness that Electron-based IDEs cannot match. The collaborative editing (CRDT-based, server-mediated) is production-quality. The built-in agent uses the Zed inference API (Anthropic-backed, via Zed's relay) and is integrated with no extension-installation friction — it ships in the box. Zed's WASM-sandboxed extension model (language extensions, theme extensions, slash commands, context providers) is more principled than VS Code's Node.js extension API and is studied as a reference for HIDE's plugin sandbox (Ch.01 §7). The language server integration (LSP, tree-sitter, syntax themes) is first-class and fast. The multi-buffer editing model (Zed can open related files from multiple repos in a unified view) is a UX differentiator.

**Where HIDE exceeds them.** Zed's inference is cloud-only (Anthropic API via Zed's relay), with no local model support planned or announced. There is no offline mode, no grammar-constrained decode, no durable memory, no living index daemon, no parallel agents, no thermal governor, and no personalization flywheel. Its plugin model, while sandboxed, is narrower than HIDE's: Zed plugins contribute language extensions, theme extensions, and slash commands; HIDE plugins contribute tools, agents, model providers, indexers, memory stores, panels, and commands (Ch.01 §7.2). Zed's multi-buffer is a view model, not an agent-aware context model; HIDE's Context Compiler (Ch.04) assembles a semantically ranked, token-budgeted context from the living index, not a flat multi-buffer.

**What HIDE harvests.** STUDY ONLY. Zed's GPUI framework architecture (retained-mode GPU rendering, text layout pipeline, CRDT-based buffer model, incremental parse tree updates) is studied as a reference for HIDE's editor surface (Ch.07) — specifically for understanding what sub-millisecond rendering requires and how to structure the view model. Zed's WASM plugin sandbox (WASM Component Model, WIT interfaces, capability-gated host functions) is studied for HIDE's plugin sandbox design (Ch.01 §7). **No Zed code is ever copied, ported, linked, or incorporated into HIDE in any form.** AGPL-3.0 requires that any derivative work be published under AGPL-3.0 — that is incompatible with HIDE's proprietary license. This is a hard rule with no exceptions, regardless of how small the snippet. If a future contributor opens a PR containing code whose structure resembles GPUI, it must be rejected and rewritten from scratch.

**Gap we close at which milestone.** N/A — Zed is study-only. HIDE's editor surface (Ch.07) is built on Tauri 2 + Monaco/CodeMirror, not Zed's GPUI. Raw editor rendering latency is not the primary focus; correctness, agent-IDE integration, and the Context Stack visibility are.

---

### §12.3.11 — Goose

**What they do well.** Goose (Apache-2.0, Block) is a desktop agent with the most production-quality MCP client implementation in Rust. The `rmcp` crate (Block's open-source Rust MCP client, also published under `modelcontextprotocol/rust-sdk`) implements both stdio (newline-delimited JSON-RPC over subprocess stdin/stdout) and Streamable HTTP (single-endpoint POST with optional SSE for server push) transports, handles `MCP-Session-Id` and `MCP-Protocol-Version` headers, implements the `initialize` / `initialized` capability negotiation handshake, supports `tools/list` with pagination, and correctly maps `CallToolResult.isError` to `ok: false`. The type definitions are precise against the 2025-11-25 spec. Goose's extensible tool system via MCP is practical: users install MCP servers (local stdio processes or remote HTTP) and Goose discovers their tools. The desktop UX (Electron-based) is approachable. Goose's Rust foundation (unlike most agent tools, which are Python or Node) makes it directly relevant to HIDE's Rust codebase.

**Where HIDE exceeds them.** Goose is model-agnostic (OpenAI-compatible, no local format) and its desktop UX, while functional, is built in Electron with no Tauri IPC, no capability-gated sandbox, and no grammar-constrained decode. Goose has no durable memory, no living index, no thermal governor, no parallel agents, and no personalization flywheel. Its MCP client, while excellent, is the entire model story — the agent loop on top is thin. Goose does not enforce a permission model on MCP tool calls: any configured MCP server gets whatever it asks for. HIDE's MCP bridge (Ch.03 §4.8) subjects every MCP tool to the full HIDE permission model (Ch.03 §4.9, Ch.10 §4.6) so an MCP tool cannot exceed its declared `capabilities_grantable` ceiling, cannot exfiltrate secrets (Ch.10 §4.8), and every call is recorded in the event log (Ch.01 §4.5) with a `capability_grant_id`.

**What HIDE harvests.** MCP Rust client (port, Apache-2.0): the `rmcp` crate's transport layer — stdio subprocess spawn with JSON-RPC newline framing, Streamable HTTP transport with `MCP-Session-Id` and `MCP-Protocol-Version` headers, SSE stream parsing for server-push, `initialize` capability negotiation (`ClientCapabilities` ↔ `ServerCapabilities`), `tools/list` polling with cursor-based pagination, `tools/call` dispatch — is ported into HIDE's `mcp/` module (Ch.03 §4.8). The `rmcp` crate's type definitions (`Tool`, `CallToolResult`, `TextContent`, `ImageContent`, `EmbeddedResource`, `ListToolsResult`) are used as the starting point for HIDE's MCP bridge types. HIDE adds HIDE-specific extensions on top: `capabilities_grantable`, `risk_gate`, `provenance_label` for trust-aware tool integration. HIDE crate/module: `hide-tools/src/mcp/client.rs`. License compliance: Apache-2.0.

**Gap we close at which milestone.** [SHELL-FIRST]: the MCP bridge is a shell-first feature (Ch.03 §4.8). HIDE's bridge extends the `rmcp` base with full HIDE permission-model enforcement (Ch.03 §4.9, Ch.10 §4.6) so every MCP tool is capability-gated, not trusted by default.

---

## §12.4 — The Moat Analysis

The term "moat" is overused in software. Here it means something specific: an advantage that compounds over time, is not easily replicated by a well-funded competitor in 6–12 months, and is structurally tied to choices the competitor has already made and cannot reverse. HIDE has six such advantages; they interact.

### §12.4.0 — Overview

The six moats interact. The format moat enables the cost moat (32B fits where Q4_K_M does not → the tier is available at all). The cost moat enables the personalization moat (fine-tune runs are free → the flywheel turns continuously). The personalization moat strengthens the trust moat (training data never leaves the machine → compliance is unconditional). The trust moat drives the feedback-loop moat (security-conscious developers adopt HIDE → more training signal per user → better model). The hardware moat runs underneath all of them (Apple Silicon unified memory + Metal GPU is the physical substrate).

```
Format moat
    ↓ enables
32B in 18GB unified → cost moat (zero marginal tokens)
    ↓ enables
Personalization flywheel → trust moat (training on-device)
    ↓ strengthens                  ↓ drives adoption of
Hardware moat ←─────────────────── Feedback-loop moat
(Apple Silicon / Metal / IOReport)  (compounding per-user quality)
```

A cloud competitor that wanted to replicate this stack would need to:

1. Build a sub-4-bit trellis quantizer competitive with `.tq` for Apple Silicon (**months**).
2. Deploy it to user hardware, not their datacenters (a different product entirely — requires a desktop app and local serving).
3. Accept that training on user code on their infrastructure is prohibited by the privacy constraint that makes the trust moat work (a logical impossibility for the combined product).

No single investment closes all three. The moats are co-dependent by design.

---

### §12.4.1 — The model format moat

`.tq` is Hawking Condense's format — the absorbed strand-quant STR2 trellis-quantized representation using per-row orthogonal random Hadamard transforms and a trellis decoder. It is not GGUF, not GPTQ, not AWQ, not a format any cloud inference provider serves or any open-source runtime decodes. A cloud competitor that wanted to offer `.tq` inference would need to implement the full Condense quantization stack (the `baker`, the `TQ3`/`TQ2` level sweep, the residual-channel outlier handling), the GPU bitslice GEMV kernel (`strand_bitslice_gemv_tcb`), and adapt their serving infrastructure to a new weight layout — months of work with no existing OSS shortcut. The more `.tq` models the Condense pipeline produces and the more LoRA adapters are expressed in `.tq` format, the more the format moat compounds. User `.tq` weights, Condense-trained personalized checkpoints, and `.tq`-format adapters are all portable only within the Hawking ecosystem: they cannot be loaded into llama.cpp, Ollama, vLLM, or any other serving framework.

The specific capability the format enables — a 32B model in ~18 GB unified memory versus ~20 GB for Q4_K_M GGUF — is the margin that makes the 32B tier accessible on a 24 GB M3 Max without swapping. Swapping kills decode throughput (the bandwidth is PCIe-to-SSD, not unified memory). A model that fits without swapping decodes at ~119 tps (measured RWKV-7 flat-to-8k); a model that swaps decodes at ~5–15 tps. That 2 GB difference is the product.

The broader context: GGUF has a thriving ecosystem because llama.cpp is OSS and runs on any hardware. `.tq` has a smaller ecosystem by design — it is co-designed with Apple Silicon Metal. The ecosystem grows with each Condense-produced model: `qwen-7b.tq`, `qwen-32b.tq`, and eventually the larger tiers. Each published `.tq` model is a format moat asset that a GGUF-serving competitor cannot natively serve.

---

---

### §12.4.2 — The personalization moat

The fine-tune-at-condense flywheel (Ch.06 §4.10) captures accepted diffs — the user approves a proposed edit, and the diff is logged as a teacher-forced training example with the preceding context as the prompt. Over weeks and months of use, the local model is fine-tuned on the user's specific codebase idioms, variable naming conventions, test patterns, error-handling idioms, and refactoring preferences. This is a moat that cloud providers structurally cannot replicate: training on a user's proprietary code on cloud infrastructure exposes that code to the provider's infrastructure and potentially to future training runs or audit access. GDPR, SOC 2, and enterprise IP agreements prevent it.

The personalized model lives on the user's machine and is trained on the user's machine; the provider never sees the training data. The more a developer uses HIDE, the more the local model diverges from any generic cloud model in the user's favor — and that divergence is not portable to a competitor without re-training from scratch on the same data the competitor cannot access. After 12 months of daily use with ~30K accepted diffs, the user has a model that is a specialist in their codebase, in the same way a senior engineer who has worked in the codebase for a year is a specialist. No cloud product can offer this.

---

### §12.4.3 — The cost moat

At zero marginal cost per token, HIDE can authorize behaviors that cloud providers would never permit:

- Running 50 parallel agents on a single task to pick the best result via oracle-gated selection (Ch.09 §4.4.2). At $0.015/1K output tokens and 200K output tokens per agent, that is $150 per run at Sonnet 4 pricing. HIDE runs it for free.
- Retrying a failing build 100 times with varied strategies — different prompts, different temperature settings, different search scopes. At any per-token cost, this burns budget. At zero per-token cost, it is the oracle strategy.
- Running full-repo re-embedding nightly — embedding every changed file in the codebase, re-ranking the repo-map, updating the semantic index (Ch.05 §4.9). At cloud embedding rates ($0.0001/1K tokens on text-embedding-3-small), a 500K-token codebase re-embedded nightly costs ~$18/month. HIDE does it for free, continuously.
- Keeping a 128K-token context window open continuously for a multi-day refactor without closing the session. Cloud tools must truncate or summarize; HIDE does not.

These are not hypothetical: the event log (Ch.01 §4.5) records every step so the context can always be reconstructed, not just preserved. The cost moat is not "we're cheaper" — it is "we have removed the constraint that governs how these systems are designed."

---

### §12.4.4 — The hardware moat

HIDE's Metal GPU kernels for `.tq` decode (the `strand_bitslice_gemv_tcb` path, proven for RWKV-7 and extending to all-linear Qwen via `tq_gpu.rs`), the thermal governor reading Apple Silicon's IOReport thermal metrics, and the unified-memory model that allows model weights and the OS to share the same physical memory pool without a PCIe bus crossing — none of these translate to a cloud VM. Cloud GPU instances have discrete GPUs connected via PCIe; the bandwidth arithmetic that makes `.tq`'s decode throughput competitive with Q4_K_M GGUF on unified memory does not hold on a PCIe-attached A100 (where the bottleneck is the PCIe bus at ~64 GB/s, not the 400 GB/s unified memory bandwidth of M3 Ultra).

HIDE's hardware moat is tightly coupled to the Apple Silicon platform and intentionally so.

The platform's unified memory architecture is the physical basis of the zero-marginal-cost model:

- A 32B model that fits in unified memory without swapping has ~119 tps decode throughput (measured RWKV-7).
- A 32B model that exceeds available RAM and swaps to SSD decodes at ~5–15 tps — a 8–24× degradation.
- The `.tq` 2 GB savings versus Q4_K_M GGUF is the difference between the first case and the second on a 24 GB M3 Max.

The thermal governor (Ch.09 §4.6) reads Apple's IOReport framework: per-cluster CPU/GPU power draw, thermal pressure levels (`IOThermalLevel`), and DVFS (dynamic voltage/frequency scaling) states — an interface that does not exist on any cloud VM or Linux server. The governor uses these readings to:

1. Pre-empt lower-priority agents before thermal throttling begins (not after).
2. Adjust `max_batch_size` down when GPU power exceeds a configurable threshold.
3. Resume preempted agents when the thermal budget recovers.

No cloud tool has any of this because cloud tools run on infrastructure they do not control. HIDE is designed for hardware it knows exactly, at the register level.

---

### §12.4.5 — The trust moat

For enterprises in regulated industries — finance, healthcare, defense, legal, aerospace — code cannot leave the machine under any circumstances. This is not a preference; it is a compliance requirement enforceable under HIPAA (healthcare data in code comments or SQL schema), SOC 2 Type II (third-party data handling attestation), GDPR (EU personal data in test fixtures), ITAR/EAR (defense-related source code), and sector-specific regulations. Cloud AI coding tools are, by their nature, non-compliant with these requirements for covered codebases: every token sent to Anthropic, OpenAI, or GitHub traverses their infrastructure, is processed by their GPUs, and is potentially retained for their audit trail.

HIDE's local-first architecture means compliance is the default, not an add-on. The hash-chained append-only event log (Ch.01 §4.5, Ch.10 §4.11) provides a tamper-evident audit trail that a cloud provider's session logs cannot match: a cloud provider's log is under their control and can be modified; HIDE's log is on the user's machine, hash-chained so tampering is detectable by re-hashing the chain. For the enterprise buyer evaluating a coding AI tool, "data does not leave the machine" is not a differentiator — it is a prerequisite for evaluation. HIDE is the only coding IDE that satisfies it.

The Seatbelt sandbox (Ch.10 §4.5) is a further trust signal: the agent's shell execution is confined to the workspace, with network default-deny and an egress proxy allowlist. An enterprise security team can review the Seatbelt profile and the proxy allowlist and make a deterministic statement about what data can leave the machine. That conversation is not possible with a cloud-first tool.

---

### §12.4.6 — The feedback loop moat

The moats above compound through a feedback loop: the more a developer uses HIDE, the better the local model gets at their codebase (personalization), the more the user's workflow shifts to HIDE-native patterns (event log, context stack, parallel agents), and the more the user's personalized `.tq` checkpoint diverges from any generic model.

The compounding schedule (approximate, under regular daily use):

| Time since first use | Accepted diffs logged | Effect |
|---|---|---|
| Week 1 | ~0–100 | Model is a generic 7B or 32B `.tq`. Quality comparable to any OSS tool with a local backend. |
| Month 1 | ~2K | First fine-tune run completes. Model begins to prefer codebase naming conventions and error-handling idioms over its generic training distribution. |
| Month 3 | ~6K | Second+ fine-tune runs. Model generates test stubs matching the project's test framework automatically. Context retrieval improves as the living index has accumulated edit-frequency signal. |
| Month 6 | ~12K | Model is measurably better on the concrete task set (test-passing diffs, compilation-successful edits) than a fresh generic model at the same parameter count. |
| Year 1 | ~30K | Model is a specialist. Performance gap versus generic cloud models on codebase-specific tasks is large; the model is a senior-engineer-level codebase expert for that specific repository. |

Cloud models are stateless per-session and cannot accumulate this signal.

The privacy constraint is why this moat is exclusive to local: training on user code on cloud infrastructure means the provider holds the training data, exposes it to their audit trail, and risks it appearing in future model outputs. GDPR Article 22 (automated decision-making using personal data), SOC 2's data-handling attestation requirements, and enterprise IP agreements all preclude it. The feedback loop moat works *because* it is local; the trust moat is the reason it stays local.

The feedback loop moat is the same dynamic that made Google Search better than competitors as query volume grew — the feedback loop between use and quality is local to the user and inaccessible to competitors without that user's training data. Unlike search, where Google could observe all queries, HIDE's training data never leaves the machine. That is the stronger version of the moat.

---

## §12.8 — Open Questions and Competitive Risks

No competitive analysis is honest without naming where the strategy could fail. This section lists the real risks in order of severity.

### §12.8.1 — Quality gap on hard reasoning tasks

**The risk:** Frontier cloud models (Sonnet 4, Opus 4, o3) are substantially ahead of any 7B or 32B local model on the hardest reasoning tasks — adversarial multi-file refactoring, subtle bug diagnosis, architecture-level planning over large codebases. A developer whose primary need is this class of task will choose Claude Code, full stop.

**The mitigation:** Grammar-constrained tool calls and oracle-gated verification (Ch.02, Ch.06 §4.5) improve *reliability* on tool-heavy tasks even at smaller parameter counts. The 32B `.tq` model tier (runtime-testing) closes the gap for a large class of practical coding tasks. The personalization flywheel (Ch.06 §4.10) further narrows the gap on a per-user, per-codebase basis over time.

**The honest assessment:** For pure reasoning quality today, HIDE at 7B is not competitive with Sonnet 4. HIDE at 32B `.tq` is competitive on tool-heavy, constrained, checkable tasks. The gap is real and must be communicated honestly to users.

---

### §12.8.2 — Local latency gap at smaller parameter counts

**The risk:** A Cursor user on an M3 Mac gets tab-complete at ~50ms from Cursor's cloud (a proprietary fast model, likely faster than HIDE's 7B local model). HIDE's 7B local complete is ~80–200ms (measured; the `.tq` GPU path narrows this but does not eliminate it at 7B). For a user whose primary metric is tab-complete latency, this is a real degradation.

**The mitigation:** The speculative decode path (Ch.06 §4.8) uses the RWKV-7 or fast-draft proposer to reduce effective latency. Grammar-guided prefix reduces decoding steps for constrained outputs. At 32B with the GPU bitslice GEMV path (staged), decode throughput measured at ~119 tps (flat to 8K context) — matching or exceeding cloud latency at longer context windows.

**The honest assessment:** At 7B, tab-complete latency is comparable, not better. The latency advantage reverses at longer context windows (RWKV-7 flat 118–119 tps to 8K vs. Qwen transformer 40→8.6 tps measured, ~14× at 8K). For short completions from a cold start, cloud is faster today.

---

### §12.8.3 — Platform monoculture risk

**The risk:** HIDE is Apple Silicon macOS only. The Metal GPU kernels, the Seatbelt sandbox, the IOReport thermal governor, and the Tauri 2 native integration are all platform-specific. A developer on Windows or Linux cannot use HIDE.

**The mitigation:** Apple Silicon is the dominant platform for development-focused laptops (Mac market share in the developer segment is >50% by several survey estimates). The Mac Studio and Mac mini are the target "workstation" hardware. The platform monoculture is a feature, not a bug: tight coupling to Apple Silicon is the physical basis of the cost model and the hardware moat. A cross-platform HIDE would be a worse product.

**The honest assessment:** HIDE addresses a large but not universal market. Windows and Linux developers are not addressable. The tight platform coupling is intentional but must be acknowledged as a constraint.

---

### §12.8.4 — The `.tq` ecosystem is small

**The risk:** `.tq` models must be produced by Hawking Condense. The community that produces GGUF models (llama.cpp, Ollama, LM Studio) is orders of magnitude larger than the Hawking Condense user base. The HIDE user cannot use a newly released GGUF model on day one; they must wait for a Condense-produced `.tq` version.

**The mitigation:** The model-provider abstraction (Ch.01 §4.3) allows HIDE to fall back to any OpenAI-compatible backend, including a local llama.cpp or Ollama serving GGUF. The `.tq` format is the *preferred* path for performance; it is not the *only* path. The fallback degrades the performance properties (no grammar masks from the runtime, no logit readback) but preserves the IDE functionality.

**The honest assessment:** The model catalog for `.tq` is small today. HIDE ships with a localhost HTTP surface that accepts GGUF models via Ollama for the initial release; the `.tq` path is the production target.

---

### §12.8.5 — The personalization flywheel is Tier-2

**The risk:** The fine-tune-at-condense flywheel (Ch.06 §4.10) is the single most compelling moat claim in this chapter. It is a [TIER-2 / PLANNED] feature, not a shell-first feature. A developer evaluating HIDE at shell-first launch will not see this capability in action.

**The mitigation:** The data-capture seam (recording accepted diffs in the event log) is [SHELL-FIRST]. The Condense trainer exists in `tools/training/`. The Tier-2 milestone is not building the capability from scratch — it is wiring the data pipeline to the trainer. The moat begins building from day one of use; the first fine-tune run materializes at Tier-2.

**The honest assessment:** At shell-first launch, the personalization moat is a future claim, not a present reality. The competitive advantage at launch is local inference, air-gap, grammar-constrained tool calls, durable memory, and context stack visibility — all of which are shell-first features.

---

### §12.8.6 — Closed-source in a market that values open-source

**The risk:** Most of the tools in this chapter's competitive set are open-source. The developer community has a strong preference for open-source tools, especially for infrastructure that touches their codebase. HIDE's proprietary runtime and quantizer may create adoption friction.

**The mitigation:** HIDE harvests aggressively from OSS (§12.5) and contributes back where possible. The IDE shell (Tauri + TypeScript/React front-end) is a candidate for open-source publication; the Hawking runtime and Condense quantizer are the proprietary core. The OSS/proprietary boundary is drawn at the runtime, not at the IDE surface. The user's data (event log, model weights, personalized checkpoints) is 100% locally owned and exportable.

**The honest assessment:** HIDE is proprietary at the core. This is a real friction point with some developer segments. The value proposition must be strong enough that users accept it — as they have accepted Cursor's proprietary model despite VS Code being open-source.

---

## §12.5 — OSS Harvest Map (consolidated)

This table is the canonical record of all OSS components that HIDE incorporates by port or study. Every entry is tracked for license compliance. The mode column distinguishes:

- **Port**: source code from the OSS project is adapted and incorporated into HIDE's codebase. The original copyright notice must appear in HIDE's NOTICE file and in the source file header.
- **Study**: the OSS design is referenced and the implementation is original. No source lines are copied. No NOTICE entry is required, but the reference is documented for attribution and audit trail.

| Source | License | What we harvest | Mode | HIDE crate/module |
|---|---|---|---|---|
| **Cline** (Cline Bot, Inc.) | MIT | `replace_in_file` tiered matching: exact → whitespace-normalized → fuzzy → CONFLICT-with-hint; normalization of Cline `------- SEARCH` / `+++++++ REPLACE` and Aider `<<<<<<< SEARCH` / `>>>>>>> REPLACE` fence formats to the structured `edits[]` wire type | Port | `hide-tools/src/edit/apply.rs` |
| **Aider** (Paul Gauthier) | Apache-2.0 | Repo-map algorithm: tree-sitter parse → symbol/reference extraction → PageRank over dependency graph → token-budgeted selection. Extended with call-graph edges, test-coverage signal, edit-frequency weighting, and hybrid re-rank. | Port (extended) | `hide-index/src/repo_map.rs` |
| **OpenHands** (All Hands AI) | MIT | EventStream event/action/observation schema: `Action`/`Observation` typed union, `cause`-link from observation to action, `source` tagging (agent/user/environment/tool) | Port (schema) | `hide-core/src/event.rs` |
| **Goose / rmcp** (Block, Inc.) | Apache-2.0 | MCP Rust client: stdio + Streamable HTTP transports, `MCP-Session-Id` / `MCP-Protocol-Version` headers, SSE stream parsing, `initialize` capability negotiation, `tools/list` with cursor pagination, `tools/call` dispatch; core MCP type definitions | Port | `hide-tools/src/mcp/client.rs` |
| **Void** (Void Editor Contributors) | Apache-2.0 | Monaco diff UX: hunk-level accept/reject controls, ghost-text rendering for streaming suggestions, `DiffEditor` wrapper; dock/panel collapsible layout with stored widths and split-view config | Port (UI) | `hide-ui/src/diff_view.rs`, `hide-ui/src/layout.rs` |
| **Continue** (Continue Dev, Inc.) | Apache-2.0 | Retrieval glue: function-boundary chunking with tree-sitter split points, FIM template construction (prefix/suffix/filename triple), lexical BM25 + embedding cosine + cross-encoder re-rank query pipeline | Port (algorithm) | `hide-index/src/retrieval.rs` |
| **OpenCode** (OpenCode Contributors) | MIT | Session list/resume/export UX pattern; plan/act step-through interaction model (numbered steps, step-level user confirmation) | Study | Ch.07 UX design; Ch.02 §4.3 autonomy levels |
| **Kilo Code** (Kilo Code Contributors) | Apache-2.0 | Checkpoint/undo shadow-git: per-agent-run shadow git branch that snapshots workspace state before execution, enabling per-run undo without polluting the user's git log, with an index of snapshot → session event range | Study → Port | `hide-tools/src/checkpoint.rs` |
| **Zed** (Zed Industries) | AGPL-3.0 | GPUI retained-mode rendering architecture; WASM Component plugin sandbox model with WIT interfaces and capability-gated host functions | **Study ONLY** — no code copied (AGPL-3.0 incompatible with proprietary product) | N/A |

### License compliance notes

**MIT (Cline, OpenHands, OpenCode).** Requires preservation of the copyright notice. HIDE's NOTICE file will include the original copyright headers for Cline and OpenHands for any ported files. OpenCode is study-only.

**Apache-2.0 (Aider, Goose/rmcp, Void, Continue, Kilo Code).** Requires preservation of the copyright notice and a NOTICE file. HIDE's NOTICE file will enumerate each Apache-2.0 component, its version, and the source file(s) that incorporate it. Modifications to Apache-2.0 code must be stated as modifications. Each ported file will carry a header comment identifying the origin, the original license, and the nature of the modification.

**AGPL-3.0 (Zed).** No port. No contact with Zed source code. The study designation means only published documentation, blog posts, conference talks, and the observable behavior of the running editor are referenced. Future contributors are required to confirm their implementation does not draw on Zed's source.

The NOTICE file is generated at build time from a structured `harvest.toml` manifest in the repo root that encodes each component, its license, its version, and the list of HIDE source files that incorporate it. A CI job fails if any `hide-tools/src/mcp/` or `hide-index/src/` source file lacks the required origin header comment.

---

## §12.6 — When to Use Each Tool (Honest Decision Guide)

This section is written for a developer evaluating HIDE against alternatives, not for marketing. It describes the genuine use-case fit of each tool, including where HIDE is not the right answer.

### Use Claude Code when:

- You need the best raw reasoning quality on the hardest architectural or algorithmic problems today. Sonnet 4 and Opus 4 are the strongest models available for multi-file refactoring, subtle bug root-cause, and large-scale API migration. A 7B or 32B local model will not match them on these tasks in 2026.
- You are not working with sensitive/proprietary code and have no compliance constraints. The quality-per-dollar trade-off is strong.
- You work primarily in the terminal and prefer Unix composability over a GUI.
- You need to process a 200K+ token context in a single call (e.g., entire repository in context). HIDE's context window is runtime-bounded; cloud models offer larger managed windows today.

### Use Cursor when:

- You want zero migration cost from VS Code. You already have VS Code extensions, keybindings, and settings; you want AI added on top.
- Tab-complete latency and quality in the editor are the primary concern and you are willing to pay for cloud inference.
- You do not have air-gap or compliance requirements.
- The background agent (cloud-run PR generation) is the workflow you want.

### Use Aider when:

- You need a command-line tool that is model-agnostic and works with any API or local backend. Aider's `--model` flag works with hundreds of providers.
- You want git integration (auto-commit, structured commit messages) as a first-class feature.
- The repo-map quality for large codebases is the primary differentiator. Aider's repo-map is the best in class for OSS CLI tools.
- You want the auto-model leaderboard to choose the edit format for your specific model.

### Use OpenHands when:

- You want the highest SWE-bench score of any OSS agent on complex multi-step tasks.
- You are comfortable running Docker and accept the container-per-session overhead.
- You want a CodeAct-style agent that writes Python to act (versus explicit tool-calling).
- You are building a research agent or evaluation harness and want the event-stream architecture for logging.

### Use HIDE when:

- Air-gap, no-data-egress, or compliance is a requirement. HIDE is the only tool in this list where the answer is unconditional yes.
- Zero marginal cost is the design constraint. You want to run 50 agents overnight, retry 100 times, embed the full repo nightly, all for free.
- You want a local model that improves at your specific codebase over time (personalization flywheel, Tier-2).
- You want decode-level control: grammar-constrained tool calls, custom samplers, logit-guided confidence gating.
- You are on Apple Silicon and the `.tq` 32B model tier (32B fits in ~18 GB unified memory where Q4_K_M does not) is the quality level you need.
- Durable session memory — the agent remembers everything from every session, queryable, never reset — is a requirement.
- You need 24/7 autonomous overnight agents (worktree-isolated, oracle-gated, resumable) running in the background.

### HIDE is not the right answer today when:

- Raw reasoning quality on the hardest tasks is the sole criterion and you are willing to pay for cloud inference. The 32B `.tq` model will narrow this gap significantly but it is runtime-testing, not in the shell-first release.
- You need a JetBrains IDE integration (Continue supports JetBrains; HIDE is Tauri-based).
- Your team is already deeply invested in GitHub Copilot Enterprise with SSO, audit logging through GitHub Advanced Security, and GitHub-native PR workflows. The migration cost is real.
- You need Windows or Linux support. HIDE is Apple Silicon macOS only; the Tauri + Metal GPU + Seatbelt + IOReport stack is platform-specific by design.

---

## §12.7 — Quantitative Comparison: Cost at Scale

To make the cost moat concrete, this table computes the monthly API cost for a developer using a coding agent at representative usage levels. "Heavy use" is defined as 2M output tokens/day across all sessions and agents.

| Usage level | Output tokens/day | Claude Code (Sonnet 4 API, $0.015/1K) | Cursor Pro ($20/month) | HIDE (local, amortized) |
|---|---|---|---|---|
| Light (1 session, 10 tool calls) | ~5K | $2.25/day → $67.50/mo | $20/mo flat | ~$0 (power ~$0.01/day) |
| Medium (3–5 sessions, 40 tool calls) | ~50K | $22.50/day → $675/mo | $20/mo + overages | ~$0 |
| Heavy (parallel agents, 20+ runs) | ~500K | $225/day → $6,750/mo | Not supported at scale | ~$0 |
| HIDE overnight swarm (50 agents × 200K toks) | ~10M | $2,250/night run | Not feasible | ~$0 |

Notes:

- Cursor Pro includes ~fast premium requests/month; overages are ~$0.04/request beyond the limit.
- "Power ~$0.01/day" is the incremental electricity cost of running an M3 Max at moderate load (decode ~4W GPU draw per measured session); at $0.15/kWh, 8 hours × 4W = $0.005.
- The "overnight swarm" row is the decisive one: HIDE enables a class of usage that is economically unthinkable with cloud inference and becomes routine at zero marginal cost.

This is not a price war. It is a fundamental change in what behaviors are rational to implement. When a behavior costs $0 to attempt and $0 to retry, the rational strategy is to attempt it exhaustively — which is exactly what Ch.09's tournament/best-of-N orchestration formalizes.

---

---

## §12.9 — Roadmap and Competitive Positioning by Milestone

This section maps HIDE's competitive positioning against the milestone structure from Ch.13. It answers: at each milestone, which competitive claims become true?

### Shell-first (M0 → M1): the foundation of the local-plane claims

At shell-first launch, HIDE ships:

- **Air-gap / local inference** — the `.tq` runtime (CPU parity oracle + staged GPU) is a localhost HTTP surface. HIDE works fully offline after model download.
- **Grammar-constrained tool calls** — `json_constrain.rs` + `JsonVocabIndex` are in-tree and working. Shell-first ships the schema-driven extension (Ch.06 §4.5). Format-failure rate for tool calls drops to ~0%.
- **Durable event log** — Ch.01 §4.5's hash-chained append-only log is shell-first. All sessions are recorded, queryable, replayable.
- **Living Index daemon** — Ch.05 §4.9's file-watching incremental indexer runs continuously. Retrieval latency is index-lookup, not re-parse.
- **Context Stack panel** — Ch.07 shows the user exactly what is in the agent's context. Full provenance visibility.
- **Tiered applier** — Ch.03 §4.7's exact-match → fuzzy → AST-aware edit applier ships. Edit reliability is better than Cline/Aider's string-match approach.
- **MCP bridge** — Ch.03 §4.8's `rmcp`-based client is shell-first. Any MCP server works with HIDE's permission model.
- **Seatbelt sandbox** — Ch.10 §4.5's Tier-1 sandbox (Seatbelt + network-deny proxy) is shell-first.
- **OSS harvest** — All §12.5 ports (Cline, Aider, OpenHands, Goose/rmcp, Void, Continue, Kilo Code) are incorporated.

**Competitive claims that are true at M1:**
The air-gap/privacy/compliance claim is true. Grammar-constrained decode claim is true. Durable memory claim is true. Context stack visibility claim is true. Tiered applier quality claim is true.

**Competitive claims that are NOT yet true at M1:**
Personalization flywheel (Tier-2). Parallel agent swarms (Tier-2). KV-cache inter-agent handoff (Tier-2). Overnight workstation mode (Tier-3). The 32B reasoning quality gap is not closed at M1.

---

### Tier-2 (SWARM milestone): parallel agents and personalization

At Tier-2, HIDE ships:

- **Parallel agent orchestrator** — Ch.09's worktree-isolated multi-agent scheduler with resource governor, tournament/best-of-N selection, overnight batch mode. The "50 free parallel agents" claim becomes true.
- **Personalization flywheel first run** — The data-capture seam was shell-first; Tier-2 wires the Condense trainer invocation. First personalized checkpoint is produced.
- **KV-cache inter-agent handoff** — `copy_kv_prefix_to_slot` wired into the orchestrator for prefix sharing across agents.

**New competitive claims that become true at Tier-2:**
The parallel agent swarm claim. The personalization moat begins compounding. The KV handoff claim.

---

### Tier-3+ (WORKSTATION / 32B): quality leadership and workstation mode

At Tier-3 and beyond:

- **32B `.tq` GPU serving** — The GPU bitslice GEMV kernel (`strand_bitslice_gemv_tcb`) matures from staged to production. 32B model in ~18 GB unified memory with ~119 tps decode. Quality gap versus Claude Code narrows significantly on tool-heavy tasks.
- **Remote workstation mode** — Ch.09 §4.9's `wss`/JSON-RPC protocol. A laptop drives a Mac Studio running the model fleet.
- **Personalization at depth** — Multiple fine-tune rounds. The model is a specialist at the user's specific codebase.

**New competitive claims that become true at Tier-3+:**
The 32B reasoning quality gap narrows. The workstation-mode claim becomes true. The personalization moat becomes measurably compounding.

---

### The competitive milestone summary

| Claim | M1 (shell) | Tier-2 (swarm) | Tier-3 (workstation) |
|---|---|---|---|
| Air-gap / offline | ✅ | ✅ | ✅ |
| Zero per-token cost | ✅ | ✅ | ✅ |
| Grammar-constrained tool calls | ✅ | ✅ | ✅ |
| Durable event log | ✅ | ✅ | ✅ |
| Context stack visibility | ✅ | ✅ | ✅ |
| Seatbelt sandbox | ✅ | ✅ | ✅ |
| Parallel agent swarms (free) | ❌ | ✅ | ✅ |
| Personalization flywheel (first run) | ❌ | ✅ | ✅ |
| KV-cache inter-agent handoff | ❌ | ✅ | ✅ |
| 32B `.tq` GPU quality tier | ❌ | ❌ | ✅ |
| Remote workstation mode | ❌ | ❌ | ✅ |
| Personalization at depth (12+ months) | ❌ | ❌ | ✅ (ongoing) |

---

---

## §12.10 — Index of Competitive Claims by Bible Chapter

This index answers: for each competitive advantage HIDE claims, which chapter owns the implementation? Use it to cross-check that a claim is backed by a real design, not a future aspiration.

| Claim | Owner chapter | Implementation status | Depends on |
|---|---|---|---|
| Grammar-constrained tool calls (~0% format failure) | Ch.06 §4.5 | [SHELL-FIRST] | `json_constrain.rs` in-tree |
| Custom sampler profiles (per-task temperature/top-p/DRY) | Ch.06 §4.6 | [SHELL-FIRST] | Extend `SamplingParams` |
| Logit-level confidence gating | Ch.06 §4.7 | [SHELL-FIRST / RUNTIME-GATED] | Metal logit readback is gated |
| Speculative decode (Eagle5, n-gram, suffix) | Ch.06 §4.8 | In-tree (proven) | `speculate/` crates |
| LoRA hot-swap per language/task | Ch.06 §4.9 | [PLANNED] | Runtime adapter API |
| Personalization flywheel (fine-tune on accepted diffs) | Ch.06 §4.10 | [TIER-2] | Condense trainer |
| Multi-model concurrency + thermal/RAM scheduling | Ch.06 §4.11 / Ch.09 §4.6 | [TIER-2] | IOReport governor |
| Durable hash-chained event log (never reset) | Ch.01 §4.5 | [SHELL-FIRST] | SQLite WAL + hash-chain |
| Context Stack panel (user sees exactly what agent sees) | Ch.07 §4.x | [SHELL-FIRST] | Tauri IPC + Ch.04 |
| Tiered edit applier (exact → fuzzy → AST-aware) | Ch.03 §4.7 | [SHELL-FIRST] | tree-sitter Ch.05 |
| MCP bridge with permission model | Ch.03 §4.8 / Ch.10 §4.6 | [SHELL-FIRST] | rmcp harvest |
| Living Index daemon (always-fresh symbol graph) | Ch.05 §4.9 | [SHELL-FIRST] | file-watcher + tree-sitter |
| Repo-map ranking (PageRank + signals) | Ch.05 §4.6 | [SHELL-FIRST] | Aider harvest (extended) |
| Hybrid retriever (BM25 + embedding + rerank) | Ch.05 §4.7 | [SHELL-FIRST] | Continue harvest |
| Seatbelt sandbox (Tier 1, network-deny) | Ch.10 §4.5 | [SHELL-FIRST] | macOS sandbox-exec |
| Parallel agent orchestrator (worktree isolation) | Ch.09 §4.2–§4.6 | [TIER-2] | Ch.02 per-run |
| Best-of-N oracle selection | Ch.09 §4.4.2 | [TIER-2] | Ch.02 oracle trait |
| Overnight batch jobs (checkpoint/resume) | Ch.09 §4.7 | [TIER-2] | Ch.01 event log |
| KV-cache inter-agent prefix sharing | Ch.06 §4.11 / Ch.09 | [TIER-2] | `copy_kv_prefix_to_slot` |
| Remote workstation (laptop → Mac Studio) | Ch.09 §4.9 | [TIER-3] | wss/JSON-RPC protocol |
| `.tq` 32B GPU decode (~119 tps) | Ch.06 §4.2 / Condense | [RUNTIME-TESTING] | `strand_bitslice_gemv_tcb` |
| Air-gap / offline (no network after download) | Ch.01 §4.3 / Ch.10 | [SHELL-FIRST] | localhost HTTP only |
| Secrets in macOS Keychain (never in model) | Ch.10 §4.8 | [SHELL-FIRST] | Keychain API |
| Tamper-evident audit log (hash-chained) | Ch.10 §4.11 | [SHELL-FIRST] | Ch.01 event log |

---

## Cross-references

| Chapter | What this chapter draws on |
|---|---|
| Ch.01 §4.3 | Stable localhost HTTP surface (the architecture that enables air-gap) |
| Ch.01 §4.5 | Append-only hash-chained event log (persistent durable memory row, trust moat) |
| Ch.01 §7.2 | Extension manifest — the plugin-contribution surface compared against Zed's narrower model |
| Ch.02 §4.3 | Autonomy levels — plan/act step-through that OpenCode and Cline both implement |
| Ch.02 §4.6 | Oracle-gated merge — the correctness guarantee that Copilot Workspace lacks |
| Ch.03 §4.7 | Tiered applier (Cline + Aider harvest) |
| Ch.03 §4.8 | MCP bridge (Goose/rmcp harvest) |
| Ch.03 §4.9 | Tool permission model that wraps every MCP tool call |
| Ch.04 | Context Compiler — context-stack visibility credited to HIDE in §12.2 |
| Ch.05 §4.6 | Repo-map ranking algorithm (Aider harvest, extended) |
| Ch.05 §4.7 | Hybrid retriever (Continue harvest) |
| Ch.05 §4.9 | Living Index daemon — the standing always-fresh index distinguishing HIDE |
| Ch.06 §4.5 | Grammar-constrained decode service — tool-call reliability differentiator |
| Ch.06 §4.7 | Logit-level signals — unavailable to any cloud or model-agnostic tool |
| Ch.06 §4.10 | Fine-tune-at-condense personalization flywheel — the compounding moat |
| Ch.06 §4.11 | Multi-model concurrency + thermal/RAM-aware scheduling |
| Ch.07 | Editor surface — Void Monaco diff UX + dock layout harvest |
| Ch.09 §4.4 | Parallel agent tournament / best-of-N — the zero-marginal-cost swarm |
| Ch.09 §4.6 | Resource governor — hardware-coupled thermal/RAM scheduler |
| Ch.10 §4.5 | Seatbelt sandbox — native macOS isolation replacing Docker (OpenHands) |
| Ch.10 §4.6 | Capability/permission model at OS scale — wrapping every MCP call |
| Ch.10 §4.11 | Tamper-evident audit log — the enterprise trust moat |
