# Hawking IDE — Technical Bible

**13-chapter engineering reference for the HIDE local-first agentic coding IDE.**

HIDE is built on Hawking Condense (`.tq` sub-4-bit models) and runs entirely on Apple Silicon — no cloud required, no per-token cost. This bible is the single authoritative source for every architecture decision, schema, state machine, and integration contract. Read it before touching any HIDE crate.

---

## Reading order

Start with [ch.00](00-vision-and-constitution.md) (the thesis and design constitution) and [ch.01](01-system-architecture.md) (process model and event log). Then read chapters in any order — they are heavily cross-referenced.

---

## Chapters

| # | Title | Core contract | Status |
|---|---|---|---|
| [00](00-vision-and-constitution.md) | Vision and Constitution | Thesis, local superpowers, 12 design principles, scope gates | ✅ |
| [01](01-system-architecture.md) | System Architecture | Four-tier process model, append-only event log, plugin spine, data stores | ✅ |
| [02](02-agent-kernel.md) | Agent Kernel | Formal FSM, Plan-as-data (HTN+DAG), Oracle trait, search escalation ladder, Skill Library | ✅ |
| [03](03-tool-system-and-capabilities.md) | Tool System and Capabilities | `ToolCall`/`ToolResult`/`ToolError` schemas, ~60 built-in tools, MCP 2025-11-25 host+client, PermissionPolicy | ✅ |
| [04](04-context-engineering-and-memory.md) | Context Engineering and Memory | Context Compiler, `KvStore` trait, CoALA memory layers, forever project memory | ✅ |
| [05](05-codebase-intelligence.md) | Codebase Intelligence | BLAKE3 merkle-DAG, tree-sitter + stack-graphs + headless LSP, unified graph, living index daemon | ✅ |
| [06](06-model-inference-orchestration.md) | Model Inference Orchestration | Model-role system, routing cascade, constrained/grammar decode, LoRA hot-swap, personalization flywheel | ✅ |
| [07](07-hci-ux-and-ide-surface.md) | HCI, UX and IDE Surface | Six-region workbench, Context Stack right-rail, Tauri IPC, Monaco adapters, three ASCII state machines | ✅ |
| [08](08-research-and-knowledge-lab.md) | Research and Knowledge Lab | Multi-source pipeline FSM, ~21 node knowledge graph, `SourceAdapter` trait, adversarial verify | ✅ POST-SHELL |
| [09](09-parallel-agents-and-workstation.md) | Parallel Agents and Workstation | 7 orchestration patterns, worktree isolation, merge funnel, machine Governor, remote wss protocol | ✅ POST-SHELL |
| [10](10-local-first-infrastructure-and-security.md) | Local-First Infrastructure and Security | Storage layout, T0-T4 sandbox ladder, CaMeL taint, 22-row threat model, hash-chained log | ✅ |
| [11](11-bleeding-edge-and-moonshots.md) | Bleeding-Edge Capabilities and Moonshots | Personalization flywheel, RLEF on-device, self-improving workflows, KV handoff, ranked table | ✅ |
| [12](12-competitive-matrix.md) | Competitive Matrix | HIDE vs Claude Code / Cursor / Cline / Aider / Void / OpenHands and 6 others; OSS harvest map | ✅ |
| [13](13-roadmap-and-build-sequencing.md) | Roadmap and Build Sequencing | M0/M1/M2/M3+ milestones, thesis gate, kill protocol, scope contract, HF lane conditions | ✅ |

---

## Key schemas (quick reference)

| Schema | Defined in |
|---|---|
| `Event` envelope (ULID, seq, session, cause, payload) | ch.01 §1.2 |
| `Plan` + `PlanStep` + `StepResult` | ch.02 §2.3 |
| `ToolCall` / `ToolResult` / `ToolError` | ch.03 §3.1 |
| `PermissionPolicy` (rules, defaults, risk_gates, scope_grammar) | ch.03 §3.5, ch.10 §10.4 |
| `ContextManifest` (retained/dropped spans, KV budget) | ch.04 §4.2 |
| `MemoryRecord` (CoALA-typed: working/episodic/semantic/procedural) | ch.04 §4.5 |
| `ToolCall` wire format for MCP 2025-11-25 | ch.03 §3.4 |
| `ModelDescriptor` / `ProviderCaps` / `RoleRegistry` | ch.06 §6.1 |
| `AgentJob` / `GovernorState` / `WakeReport` | ch.09 §9.5 |
| `PersonalizationRecord` | ch.11 §11.1 |
| `AgentHandoff` (structured JSON inter-agent) | ch.11 §11.4 |

---

## Scope summary

**Ships in the initial shell (M0–M1):** agent kernel, eval harness, router, `.tq` GPU serving, constrained-JSON tools, Tauri app shell (Chat / Diff Review / Context Stack / terminal / file tree / timeline).

**M2, gated on thesis-gate GO:** `hawking-condense` (HF link → .tq), `hawking-hub` (catalog download), Model Lab/Store tab.

**Explicitly deferred:** Hawking HF org live catalog (until 32B `.tq` ready), Research Tab, parallel agent workstation UI, Remote Mac-Studio, RLEF/personalization flywheel (ch.11 moonshots), full JSON-Schema grammar compiler.

See [ch.13 §13.9](13-roadmap-and-build-sequencing.md) for the full scope contract table.

---

## License policy

Only **MIT / Apache-2.0** code may be copied or ported into shipped `app/` or `crates/`. CI gate: `cargo-about` + `package.json` license check fails on GPL/AGPL/unknown. See `THIRD_PARTY_NOTICES.md` for attribution. **Zed = study-only (AGPL). Cursor/Codex = never lift (proprietary ToS).**

See [ch.12 §12.5](12-competitive-matrix.md) for the full OSS harvest map.
