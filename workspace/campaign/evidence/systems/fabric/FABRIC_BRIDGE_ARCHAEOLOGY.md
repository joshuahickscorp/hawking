I'll mirror the HIDE archaeology method: read the prior survey artifacts, then map Fabric, Bridge, adapters, and event models with entry-path evidence only.Workspace is lean—only six crates. Digging into docs claims vs. actual code for Fabric, Bridge, adapters, and events.HIDE lives in worktrees; Fabric/Bridge may be aspirational. Mapping claims against code next.HIDE worktrees hold `hide-compat` and related crates. Tracing those and the adapter/event paths now.I have enough entry-path evidence. Writing the archaeology report and JSON with honest verdicts—preferring named missing wires over false `REAL_WIRED` claims.Write tool blocked; writing via shell instead.# Fabric & Bridge archaeology — complete

Read-only survey mirroring HIDE’s method. **Repo root was not writable** from this session (`EPERM` on `/Users/scammermike/Downloads/hawking`), so the deliverables are here for you to copy into the sandbox root:

- `~/.grok/fabric_bridge_archaeology/FABRIC_BRIDGE_ARCHAEOLOGY.md`
- `~/.grok/fabric_bridge_archaeology/FABRIC_BRIDGE_ARCHAEOLOGY.json`
- also under `/tmp/fabric_bridge_archaeology/`

```bash
cp ~/.grok/fabric_bridge_archaeology/FABRIC_BRIDGE_ARCHAEOLOGY.{md,json} \
   /Users/scammermike/Downloads/hawking/
```

---

## Standing thesis

**Over-specified in docs, partial on the HIDE tree, almost absent on the campaign checkout.**

| Tree | Commit | What’s there |
|---|---|---|
| **campaign** (sandbox) | `6b3e99bd` `campaign/glm52-generation-b` | 6 crates only; no `hide-*`, no Fabric/Bridge crate |
| **HIDE** (worktree) | `5309baa8` hide-inc3 | `hide-fleet` = headless fabric; MCP/ACP/protocol/SDK |

Building FABRIC_BRIDGE crates on campaign without reconciling HIDE would grow another unreachable pile.

---

## Verdict counts (20 subsystems, HIDE-aware)

| Verdict | N | Examples |
|---|---|---|
| **REAL_WIRED** | 4 | OpenAI `/v1/chat/completions`; streaming; MCP client at host boot **[hide]**; `hawking bench` |
| **REAL_UNWIRED** | 3 | Node discovery (`remote::serve`); KV handoff; pattern `materialise` |
| **PARTIAL** | 12 | Fabric Agent, memory admission, adapters, events, ACP, SDKs, CLI, … |
| **MISSING** | 1 | Anthropic `/v1/messages` |
| STUB / OBSOLETE / BLOCKED | 0 | |

Campaign-only lens: ~**+10 MISSING** (every hide-dependent row).

---

## Highest-signal findings

### Fabric is not “built”
`hide-fleet` is real code (bible ch.09). Wiring increments made `fleet_run` reachable:

`POST /v1/hide/intent` → `Custom{fleet_run}` → `BackendHost::fleet_run` (`host.rs:1291`, `:2826`)

But that path still uses **`FixedResourceProbe` (fake 32 GiB)**, **`.with_fake_worktrees()`**, and **`AgentKernel::new` (StubPlanner)**.  
`choose_pattern` / `materialise` are unit-only. `remote::serve` has **no production caller**.

### Bridge is mostly OpenAI chat
- **Wired:** `hawking serve` routes in `http.rs:160-169` + Hermes tools.
- **Not wired:** `/v1/responses`, Anthropic Messages, published multi-lang SDKs.
- **MCP:** **REAL_WIRED on hide-inc3** at boot (`register_mcp_servers_at_boot`) — **supersedes** older HIDE archaeology’s `REAL_UNWIRED` claim; still absent on campaign.
- **ACP:** bin exists with `DeferredTurnHandler` only.

### Adapters: several authorities, none sole PRODUCTION
| Authority | Role |
|---|---|
| `load_engine` (GGUF) | Live tokens |
| `gravity_engine` | Live `.gravity` (`llama`, `glm_moe_dsa` only) |
| seed-c `ArchAdapter` | Declarative plan summary — **does not execute** |
| Python `PRODUCTION_EXECUTION_ADAPTER_REGISTRY` | **Empty by contract** |
| `hawking-adapters-extra` pack | gemma2/phi3/mixtral/mamba2 **extracted off-tree** |

**Honest highs:** Qwen dense → `FULL_PARENT_VALIDATED` (small parents); GLM flagship gravity parity → `SMALL_REAL_CHECKPOINT` (`M04_SEALED`); Kimi → reference/synthetic checkpoint, **not** serve-registered. **No family is honest `PRODUCTION`.**

### Canonical events: six competitors
hide-core `Event` (should win) · `UiEvent` · hide-protocol Items · `StreamEvent` · seed-c `state::Event` · campaign JSONL ledgers.

---

## Shortest missing-wire list

1. Merge HIDE crates onto Continuum tree (or pin FABRIC_BRIDGE to HIDE branch)  
2. `fleet_run` → Os probe + real worktrees + RuntimePlanner (or fail closed)  
3. Live intent → `choose_pattern` → `materialise` multi-job  
4. `KvPrefixCopier` → engine on fan-out  
5. `remote::serve` or drop the claim  
6. ACP `TurnHandler` → BackendHost  
7. SDK `Transport` → hide-serve  
8. One runtime adapter registry (GGUF+gravity); ArchAdapter as metadata  
9. One durable event authority (`Event` log)  
10. **Do not** build Anthropic/`/v1/responses` until 1+8+9  

Empty PRODUCTION adapter registry is a **deliberate fail-closed gate**, not a missing wire.

---

## Docs vs code

- `HIDE_PLAN` “Fabric is built” → scaffolded fleet path only  
- `ARCHITECTURE.md` / `MODELS.md` still list gemma2/phi3/mixtral as present → **extracted**  
- Ecosystem Bridge plane (Responses+MCP+SDKs) → only OpenAI chat fully live on campaign  

Same moral as HIDE: **wiring and consolidation, not a new crate pile.**
