# HCLI Memory OS Plan (Ascension Bible §17)

**Status:** PLAN + SCAFFOLD — gated on Proto-Frankenstein offload; no live model work.  
**Scaffold code:** `crates/hawking-context/src/memory_os.rs`  
**Programme gate:** future Agent OS slice; do not block Gravity / Qwen ladder.

---

## 1. Purpose

Turn Hawking's existing memory substrate into a **Memory OS** with:

- explicit **L0–L5** retention tiers
- a unified tool surface: `memory.store|retrieve|update|consolidate|invalidate|archive|forget|explain`
- a full item schema: `source`, `timestamp`, `confidence`, `scope`, `expiry`, `supersedes`, `contradicts`, `verification state`
- the hard rule: **stale claims cannot silently become permanent truth**

This plan is gap-first. Most of the durable machinery already exists. The scaffold adds the tier model, full schema, and tool contract without rebuilding working stores.

---

## 2. Audit summary — what already exists

| Component | Location | What it already does |
|-----------|----------|----------------------|
| Six-class memory | `hawking-context::memory_classes` | Working / Episodic / SemanticProject / Procedural / User / Verification; write-authority caps; pin / expire / forget / inspect / correct / export; `supersedes`; `expire_at_ms` |
| Hierarchical store | `hawking-context::memory` | FTS5 + cosine retrieval; Generative-Agents score; `MemoryKind`; pin; decay; version / supersedes |
| Outcome ledger | `hide-backend::memory` | `source`, `confidence`, `expiry_ms`, `invalidation`, citations, quarantine on bad outcomes, revalidation, supersedes |
| File memory tool | `hide-kernel::tooling_memory` | Claude-parity path-rooted scratch (`view/create/str_replace/…`) |
| Context rot | `hawking-context::rot` | Manifest rot detection (not claim lifecycle) |
| Compiler contradictions | `hawking-context::manifest` | Surfaced span contradictions at compile time |

### Class → retention mapping (already real)

| Class | Retention today | Closest §17 tier |
|-------|-----------------|------------------|
| Working | turn-local; `end_turn` clears | **L0 ACTIVE** |
| Episodic | session; `evict_session` | **L1 SESSION** |
| SemanticProject / User / Verification | durable workspace (user cross-ws) | **L2 PROJECT** |
| Procedural | durable recipes / skills-ish | **L3 SKILLS** (pre-Foundry) |
| — | **missing as first-class** | **L4 GRAVEYARD** |
| — | **missing as first-class** | **L5 ARCHIVE** |

### Prior art — Claude auto-memory

Claude project memory files use YAML frontmatter:

```yaml
---
name: …
description: "…"
type: project | feedback | …
originSessionId: …
---
```

Useful pattern for **source + scope + session binding** metadata. Hawking already has stronger typed fields in the ledger and classed store; the Memory OS item schema formalises the same idea as:

`source`, `timestamp_ms`, `confidence`, `scope`, `expiry_ms`, `supersedes`, `contradicts`, `verification_state`.

---

## 3. Genuinely new (gaps this scaffold fills)

1. **L0–L5 as an explicit tier enum** with retention rules and default-context eligibility.  
2. **L4 GRAVEYARD** and **L5 ARCHIVE** as first-class tiers (no dedicated class tables today).  
3. **`contradicts`** as a first-class edge (compile-time contradictions exist; durable per-item edges did not).  
4. **`VerificationState`** covering unverified → proven → contradicted / invalidated / expired / archived / graveyarded.  
5. **Unified `MemoryOs` tool trait** matching bible tool names.  
6. **Stale-claim guard:** cannot `update` an invalidated/expired/archived/graveyarded item back to a truth-eligible state; must **store a superseding** item.  
7. **`memory.explain`** — structured eligibility rationale.  
8. **`memory.consolidate`** — merge N → 1 with supersession of inputs.

### Explicit non-goals of this slice

- No second durable SQLite schema replacing `ClassedMemorySystem`.  
- No live model revalidation / semantic truth.  
- No Qwen / Gravity / frankenstein paths.  
- No push/PR/remote.

---

## 4. Memory OS design

### 4.1 Tiers

```text
L0 ACTIVE      current task state + immediate tool outputs
L1 SESSION     hypotheses, plans, checkpoints, unresolved failures
L2 PROJECT     repo map, architecture decisions, accepted receipts
L3 SKILLS      verified reusable procedures (Skill Foundry admitted)
L4 GRAVEYARD   failed mechanisms, causes, reopen conditions
L5 ARCHIVE     compressed historical evidence, source-bound artifacts
```

**Default context set:** L0–L3 only, and only when `verification_state` is truth-eligible and not hard-expired.

### 4.2 Item schema

```text
id
tier
text
source
timestamp_ms
confidence          # 0.0..=1.0
scope               # PersonalScope (existing)
expiry_ms           # optional hard TTL
supersedes          # optional id
contradicts         # list of ids
verification_state
tags
pinned              # soft-expiry only; never resurrects dead claims
updated_at_ms
```

### 4.3 Tools

| Tool | Behaviour (scaffold) |
|------|----------------------|
| `memory.store` | Insert with full schema; supersede target → invalidated; contradict edges mark both Contradicted |
| `memory.retrieve` | Keyword filter + tier filter; default drops ineligible |
| `memory.update` | Patch fields; **blocks** silent promotion of stale → truth |
| `memory.consolidate` | New item from N ids; max confidence; invalidate inputs |
| `memory.invalidate` | → Invalidated + reason tag; leaves default context |
| `memory.archive` | → L5 + Archived |
| `memory.forget` | Hard delete + dangling edge cleanup |
| `memory.explain` | Eligibility report for audit / meter |

Graveyard is exposed as `InMemoryMemoryOs::graveyard` (L4 + reopen condition tags); a host command can alias it later as `memory.graveyard` or `memory.invalidate --graveyard`.

### 4.4 Stale-claim rule (enforced in scaffold)

1. Hard expiry → `Expired`; excluded from default retrieve.  
2. Invalidate / supersede / contradict → not default-eligible.  
3. L4 / L5 never default-eligible.  
4. Pin does **not** override invalidation/archive/graveyard.  
5. Reactivation requires a **new** superseding store (or future explicit re-admit API), not a silent field flip to `Proven`.

### 4.5 Bridge plan (later, post-gate)

| Phase | Work |
|-------|------|
| M1 (done scaffold) | Types + `InMemoryMemoryOs` + tests |
| M2 | Adapter: `ClassedMemorySystem` ↔ `MemoryItem` (map class↔tier; evidence_tier→verification_state) |
| M3 | Adapter: `hide_backend::MemoryLedger` claims as L2 items with confidence/quarantine |
| M4 | Host tool dispatcher: bible tool names → `MemoryOs` |
| M5 | L4/L5 durable tables or tagged rows in existing SQL |
| M6 | Wire L3 admitted skills from Skill Foundry into procedural / Memory OS |

---

## 5. Scaffold inventory

| Path | Role |
|------|------|
| `crates/hawking-context/src/memory_os.rs` | Tiers, schema, trait, in-memory OS, tests |
| `crates/hawking-context/src/lib.rs` | Module + re-exports |
| This plan | Design + audit |

---

## 6. Tests (scaffold)

- Class → tier mapping  
- Store / retrieve / forget  
- Expiry blocks default retrieve  
- Cannot promote invalidated → proven via update  
- Contradict marks both ineligible  
- Archive + graveyard leave default context  
- Consolidate supersedes inputs  
- Explain surfaces full schema  
- All bible tools callable on trait surface  

---

## 7. Remaining work

1. Durable bridge to classed SQL + host ledger (M2–M3).  
2. Host/`hide-kernel` tool surface with bible command names.  
3. L4 reopen-condition workflow (when conditions met → propose superseding L1/L2 item).  
4. L5 compression / evidence packing (content-addressed archive blobs).  
5. Integration with Skill Foundry admission → L3 write under ProceduralWriteCap-equivalent.  
6. Optional: project Claude-style frontmatter files as an export lens (not a second authority).  

---

## 8. Confidence

| Claim | Confidence |
|-------|------------|
| Existing classed + ledger stores cover most of §17 durability | **High** |
| L4/L5 + contradicts + unified tools were the real gaps | **High** |
| Scaffold enforces stale-claim rule for the in-memory OS | **High** (tested) |
| Production bridge without dual-write bugs | **Medium** until M2–M3 lands |
| No need to rebuild ClassedMemorySystem | **High** |
