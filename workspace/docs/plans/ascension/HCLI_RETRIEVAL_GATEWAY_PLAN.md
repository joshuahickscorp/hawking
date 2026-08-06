# HCLI Retrieval Gateway Plan

**Status:** plan + scaffold (no live model / no Qwen / no Gravity)  
**Date:** 2026-08-06  
**Bible:** HAWKING_ASCENSION_BIBLE §15 (Agent retrieval gateway)  
**Gate:** future programme — after Proto-Frankenstein sealed, offloaded, hash-verified, and removed from the active local storage envelope (bible §0)  
**Scaffold crate:** `crates/hide-gateway` (`hide_gateway::retrieval`)

---

## Intent

Maintain **separate** indexes and retrieve the **smallest relevant evidence set**, never “everything the agent might need.” Every hit carries a provenance envelope so claims can be audited and critical claims can be cross-checked on a different retrieval path.

```text
WEB INDEX          papers, documentation, release notes, source repositories
REPOSITORY INDEX   files, symbols, commits, receipts, prior experiments
TOOL INDEX         MCP/HCLI tools and compatible sets
EXPERIENCE INDEX   past failures, accepted fixes, benchmark mechanisms
SKILL INDEX        verified reusable workflows
```

---

## Existing patterns reused (do not reinvent)

| Existing | Role in this gateway |
|---|---|
| `hawking-index` hybrid legs + RRF + `SearchResultSource` | REPOSITORY backend; channel identity (`Lexical` / `Symbol` / `Semantic` / `Graph`) |
| `hawking-index::semantic` RRF + rerank | Global multi-domain fusion later; scaffold ranker is deterministic lexical×authority |
| `hide-kernel::extension_registry` progressive disclosure | TOOL index compact rows first; full schema only on explicit load/grant |
| `hide-kernel::skills::SkillStore` | SKILL ranking = relevance × importance × success rate |
| ToolSearch deferred-tool pattern (session tooling) | Never inject every tool schema into the prompt |
| Agent `subagent_type` / capability modes | Domain allow-lists per role (explore → REPOSITORY+WEB; implement → +TOOL) |
| grok-orchestration MCP profile tiers | Profile-scoped domain policy (sandbox vs `gate` web) |
| `hide-core` content hashes / CAS blake3 | `ContentHash` on every hit |
| Negative-science inheritance (bible §32) + receipts | EXPERIENCE index long-term backing |

Scaffold deliberately depends only on `serde` / `blake3` / `thiserror` so it compiles without dragging hide-backend or live indexes.

---

## Typed index schemas (scaffolded)

Concrete Rust types in `hide_gateway::retrieval`:

| Domain | Record type | Key fields |
|---|---|---|
| WEB | `WebRecord` | `url`, `title`, `body`, `source_domain`, `authority_rank`, `injection_status` |
| REPOSITORY | `RepositoryRecord` | `path`, `symbol?`, `body`, `commit?`, `source_domain` |
| TOOL | `ToolIndexRecord` | `tool_name`, `description`, `schema_digest`, `version`, `effects[]` |
| EXPERIENCE | `ExperienceRecord` | `summary`, `failure_tag?`, `fix_tag?` |
| SKILL | `SkillIndexRecord` | `name`, `trigger`, `body`, `success_count`, `fail_count`, `importance` |

### Hit envelope (required on every result — bible §15)

```text
source-domain identity          SourceDomainId
retrieval-channel identity      RetrievalChannel
content hash                    ContentHash (blake3 hex)
authority rank                  AuthorityRank ∈ [0,1]
claim-to-source graph           Vec<ClaimEdge>
independent-source diversity    f32 on RankedSet + each hit
prompt-injection status         InjectionStatus { Clean | Suspected | Blocked }
```

### Ranking interface

```text
trait DomainIndex { domain(); search(query) -> Vec<RetrievalHit> }
trait RetrievalRanker { rank(query, candidates) -> RankedSet }

RetrievalGateway
  register(DomainIndex)
  retrieve(RetrievalQuery) -> RankedSet          // smallest relevant set
  cross_check_critical(claim, primary, alternate) // different path required
```

Default ranker: `MinimalSetRanker` — relevance + authority, injection penalty, hard `limit`. Diversity computed from distinct `source_domain` values among returned hits.

---

## Critical-claim cross-check

Rule: **critical claims require a different retrieval path.**

Scaffold definition of “different path”:

1. **Different `IndexDomain`** (e.g. REPOSITORY claim re-checked via WEB), or later  
2. **Different `RetrievalChannel` inside one domain** (lexical primary, semantic alternate — wire to `hawking-index` hybrid legs).

`CrossCheckReport` records both channels, alternate hits (stamped `RetrievalChannel::CrossCheck`), diversity, and a coarse `corroborated` flag.

---

## Scaffolded vs implemented

| Piece | Status |
|---|---|
| Five domain record types + `DomainIndex` trait | **Scaffolded** (`InMemoryDomainIndex`) |
| Hit envelope with all §15 fields | **Scaffolded** |
| `RetrievalRanker` + `MinimalSetRanker` + limit | **Scaffolded** + unit tests |
| `RetrievalGateway::retrieve` multi-domain | **Scaffolded** |
| `cross_check_critical` different domain path | **Scaffolded** |
| Live WEB crawler / doc ingest | **Not implemented** |
| Wire REPOSITORY → `SqliteCodeIndex` / hybrid RRF | **Not implemented** |
| Wire TOOL → `ToolRegistry` + MCP descriptors | **Not implemented** |
| Wire SKILL → `SkillStore` durable dir | **Not implemented** |
| Wire EXPERIENCE → receipts / campaign evidence | **Not implemented** |
| Model-facing packer (context compiler integration) | **Not implemented** |
| Real prompt-injection classifier (ML / rules beyond fixtures) | **Not implemented** (status field + fixture flags only) |
| ANN / vector backend for WEB | **Not implemented** (and not required for this lane) |

---

## Integration map (post–Proto-Frankenstein)

```text
hide-gateway::retrieval
    │
    ├─ REPOSITORY ──► hawking-index::{SqliteCodeIndex, HybridRetriever}
    ├─ TOOL ────────► hide-core::ToolRegistry
    │                  hide-kernel::extension_registry (compact)
    │                  hide-kernel::tooling::mcp descriptors
    ├─ SKILL ───────► hide-kernel::skills::SkillStore
    ├─ EXPERIENCE ──► receipts/ + workspace/campaign/evidence + §32 ledger
    └─ WEB ─────────► future fetch broker (bible §7 credential broker; Option-C)

consume via hawking-context::RetrievalMode::RetrieveThenPack
         + hide-backend host packing (do not dump full schemas)
```

---

## Phased delivery (when gate opens)

1. **P0 — contracts freeze** (this lane): types, ranker trait, tests, plan.  
2. **P1 — REPOSITORY adapter**: `DomainIndex` impl over `hawking-index` search; preserve `SearchResultSource` → `RetrievalChannel`.  
3. **P2 — TOOL + SKILL adapters**: compact catalog from extension registry + SkillStore; enforce progressive disclosure.  
4. **P3 — EXPERIENCE ingest**: sealed receipt summarizer → experience records (no model required for schema path).  
5. **P4 — WEB + injection**: credential-brokered fetch; real injection heuristics; critical-claim dual-path required for TG claims.  
6. **P5 — packer**: smallest-set injection into HCLI context with diversity floor for critical claims.

---

## Non-goals (this lane)

- No Qwen / Gravity / downloads  
- No frankenstein operator or evidence edits  
- No push / PR / remote  
- No detached daemons  
- No committing a venv  

---

## Acceptance for scaffold

- `cargo test -p hide-gateway` passes  
- Tests prove: separate domains, envelope fields, limit < candidates, cross-check different domain, injection not silently trusted  
- Plans live under `workspace/docs/plans/ascension/`
