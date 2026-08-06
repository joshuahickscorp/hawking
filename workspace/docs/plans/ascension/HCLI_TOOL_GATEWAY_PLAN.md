# HCLI Tool Gateway Plan

**Status:** plan + scaffold (no live model / no Qwen / no Gravity)  
**Date:** 2026-08-06  
**Bible:** HAWKING_ASCENSION_BIBLE §16 (Tool gateway)  
**Gate:** same as §15 — after Proto-Frankenstein offload (bible §0)  
**Scaffold crate:** `crates/hide-gateway` (`hide_gateway::tools`)

---

## Intent

The tool gateway retrieves **mutually useful tool sets**, not only isolated tools. It enforces credentials, policy, session affinity, effect boundaries, tool health, version, and I/O schemas. **Tool defects are classified separately from model failures** so self-TG and eval never pollute model scores with broken tooling.

### Bible kernel bundle example (scaffolded as `kernel_bundle()`)

```text
source reader        → fs.read
profiler             → profiler.sample
compiler             → compiler.invoke
benchmark runner     → bench.run
receipt verifier     → receipt.verify
artifact inspector   → artifact.inspect
```

---

## Existing patterns reused (do not reinvent)

| Existing | Role in this gateway |
|---|---|
| `hide-core::tool::{ToolSpec, ToolCall, ToolResult, ToolAnnotations}` | Catalog row shape; `ok:true` + nonzero exit is **data** |
| `hide-core::types::EffectKind` / `EffectSet` | Effect boundaries (`Read`/`Write`/`Execute`/`Network`/…) |
| `hide-kernel::tooling::registry` builtin catalog | Initial tool inventory |
| `hide-kernel::tooling::mcp` (`McpServerDescriptor`, sticky client) | Session affinity + transport health |
| `hide-kernel::extension_registry` progressive disclosure + honest effects | Compact index; reject undeclared effects |
| `hide-kernel::tools` lint / idempotency / tool loop | Call-time validation before dispatch |
| `hide-kernel::subagent::{SubagentKind, IsolationMode}` | Role-scoped bundles (research vs implement vs verify) |
| Agent tool `subagent_type` + `capability_mode` | Profile allow-lists (`read-only` / `execute` / `all`) |
| ToolSearch deferred loading | Grant full schemas only after bundle retrieve |
| grok-orchestration MCP tiers (`sandbox` vs `gate`) | Policy profile string + network allow bit |
| Permission engine (`hide-core::permission`) | Future wire for credential + grant IDs |

Scaffold stays free of `hide-core` path deps so contracts can freeze before host integration.

---

## Design

### Tool catalog + bundles

```text
ToolRef            id, name, version, effects[], input/output schema, credential?
BundleMember       tool_id, role, required
ToolBundle         id, name, members[], mutual_affinity
kernel_bundle()    §16 example set
```

Bundles are first-class retrieval units. Isolated tool fetch remains possible later as a degenerate one-member bundle.

### Enforcement surface (bible list)

| Concern | Scaffold type / behaviour |
|---|---|
| credentials | `ToolEnforcement` session→credential map; `MissingCredential` error |
| policy | `ToolPolicy { max_effect, allow_network, require_healthy, profile }` |
| session affinity | `SessionAffinity` + gateway ledger `session_has_bundle` |
| effect boundaries | `EffectBoundary` rank vs policy; deny on exceed |
| tool health | `ToolHealth` / `ToolHealthStatus`; required unhealthy → fail grant |
| tool version | `ToolVersion` required non-empty at grant |
| input/output schemas | non-null `input_schema` required at grant; full body deferred to model |

### Failure classification (separate ledgers)

```text
FailureClass::SuccessWithData     tool ok:true (exit_code is data)
FailureClass::ToolDefect          schema, timeout, crash, unhealthy, effect breach,
                                  version, credential, transport
FailureClass::ModelFailure        wrong tool, bad args, ignored result, hallucinated tool,
                                  policy circumvention attempt
FailureClass::Mixed               both sides contributed — keep both for honest scoring
```

`classify_outcome(tool, OutcomeObservation) -> FailureClass` is pure and unit-tested.

**Important contract (from hide-core):** process tools may return `ok: true` with `exit_code != 0`. That is **not** a tool defect and **not** a model failure by itself — it is `SuccessWithData`.

---

## Scaffolded vs implemented

| Piece | Status |
|---|---|
| `ToolBundle` / `kernel_bundle()` | **Scaffolded** |
| `ToolGateway::retrieve_bundle` under policy | **Scaffolded** + tests |
| Health / credential / effect / schema gates | **Scaffolded** |
| `classify_outcome` taxonomy | **Scaffolded** + tests |
| Live registration from builtin + MCP catalogs | **Not implemented** |
| Sticky MCP session transport | **Not implemented** (affinity ledger only) |
| PermissionEngine / GrantId wiring | **Not implemented** |
| Host dispatch through `ToolLoop` | **Not implemented** |
| Eval harness debiting separate defect ledgers | **Not implemented** |
| Auto-derived bundles from co-occurrence | **Not implemented** (static kernel bundle only) |

---

## Integration map (post–Proto-Frankenstein)

```text
hide-gateway::tools::ToolGateway
    │
    ├─ catalog ◄── hide-kernel::register_builtin_tools
    │              + mcp::register_mcp_servers
    │              + extension_registry compact entries
    ├─ policy  ◄── hide-core::permission + profile (sandbox|gate|…)
    ├─ dispatch► hide-kernel::tools::runner::ToolLoop
    └─ classify► hawking-eval / self-TG receipts
                 (tool_defect_* counters ≠ model_* counters)

Bundle suggestions:
  kernel     — §16 lab set
  edit       — fs.read + edit.* + git.diff + test.run
  research   — search.text + fs.read + memory (read-only profile)
  verify     — receipt.verify + artifact.inspect + test.run
```

Retrieval gateway TOOL domain and tool gateway catalog should share identity (`tool_name` / schema digest) so “retrieve tools” and “grant bundle” do not diverge.

---

## Phased delivery (when gate opens)

1. **P0 — contracts freeze** (this lane): bundles, enforcement, classification, tests.  
2. **P1 — catalog bridge**: build `ToolRef` from `ToolSpec` + MCP annotations; health from registration results.  
3. **P2 — policy bridge**: map capability profiles and permission grants into `ToolPolicy`.  
4. **P3 — dispatch bridge**: granted bundle becomes the allow-list for `ToolLoop` for that session.  
5. **P4 — eval honesty**: self-TG / HCLI product tests (bible §33) report tool-defect rates separately.  
6. **P5 — learned bundles**: co-occurrence from successful runs → candidate bundles (still policy-gated).

---

## Non-goals (this lane)

- No Qwen / Gravity / downloads  
- No frankenstein operator or evidence edits  
- No push / PR / remote  
- No detached daemons  
- No live MCP process spawning in tests  
- No committing a venv  

---

## Acceptance for scaffold

- `cargo test -p hide-gateway` passes  
- Kernel bundle exposes the six bible roles  
- Policy/health/credential denials covered  
- Schema timeout effect wrong-tool nonzero-exit classification covered  
- Plans live under `workspace/docs/plans/ascension/`
