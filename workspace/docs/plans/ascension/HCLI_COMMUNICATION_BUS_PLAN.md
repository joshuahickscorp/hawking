# HCLI Communication Bus Plan

**Status:** PLAN + SCAFFOLD ONLY — gated on Proto-Frankenstein offload.  
**Authority:** `HAWKING_ASCENSION_BIBLE.md` §20 (Communication bus).  
**Scaffold crate:** `crates/hawking-comms`  
**Schema markers:** `hcli.comms.v0`, `hcli.comms.text.v0`, `hcli.comms.structured.v0`, `hcli.comms.latent.v0`  
**Experimental gate id:** `hcli.comms.latent.experimental`

---

## 1. Purpose

Define a **three-level communication bus** so HCLI agents, sessions, and (later)
same-model Qwen peers can exchange:

| Level | Name | Character |
|-------|------|-----------|
| **1** | TEXT | Portable and inspectable natural language |
| **2** | STRUCTURED STATE | Plans, evidence graphs, typed beliefs, tool results |
| **3** | LATENT | Hidden-state / embedding / KV — **experimental only** |

LEVEL 3 in this stage is a **sealed-packet format only**. There is **no** live
cross-session latent or KV transfer implementation. The bible forbids unsealed
latent/KV transfer and requires quality, latency, security, and auditable
visible commitments before latent is treated as production.

---

## 2. Preconditions / gates

| Gate | Requirement |
|------|-------------|
| Proto-Frankenstein | Offloaded / sealed out of active envelope before Agent OS activation uses this bus in product paths |
| Same-model L3 start | First latent experiments only `Qwen 30B session → Qwen 30B session` |
| Cross-model L3 | Requires trained alignment **and** independent evidence — default **refuse** |
| Live KV bind | Deferred until experimental gate is explicitly opened by controller policy |

---

## 3. Existing patterns reused

| Existing | Reuse |
|----------|--------|
| `hide-backend::hcli_bridge` | JSONL machine-control envelope, `session` method (create/resume/list/close), capability sections, strict validation — **model for bus envelopes and session ids** |
| `hide_protocol::model::{Session, StateCapsuleRef}` | Session identity; capsule **digest pins** (not live tensors) — L3 payload refs follow the same integrity idea |
| `hide_protocol::plan::{Plan, Goal, PlanStep}` | LEVEL 2 plan payloads serialize as JSON so they can carry protocol plans without a hard dep |
| `hawking-research` CAS / claim graphs | LEVEL 2 evidence graph nodes/edges; content hashes on evidence |
| `hide_backend::lenses_evidence::EvidenceTier` | Belief/claim strength living beside structured evidence |
| `hawking-events` / `hide_core::Event` | Future durable log projection of bus packets (not wired in P0) |
| `hawking-serve::system_kv_bank` | **In-process** inference KV — **not** a cross-session latent bus; do not conflate |

**Audit note:** `hcli_bridge.rs` is transport-neutral JSONL control (capabilities,
generate, agent, swarm, session, status). It does **not** yet carry multi-level
agent-to-agent state. This plan keeps `hawking-comms` as the schema authority
for bus packets; a later phase may add bridge methods or project packets into
events without inventing a second session model.

---

## 4. LEVEL 1 — TEXT

### Shape

`TextMessage`:

- `schema`, `session_id`, `sender`, optional `recipient`
- `role` ∈ {user, agent, system, tool, peer}
- `text`, `created_unix_ms`, `content_hash` (blake3 of text)

### Properties

- Portable across models and processes  
- Fully inspectable in logs / UI / audits  
- Default inter-agent language when structure is unnecessary  

### Validation

Non-empty session, sender, text; content_hash matches body when set.

---

## 5. LEVEL 2 — STRUCTURED STATE

### Kinds

| Kind | Payload | Notes |
|------|---------|--------|
| `plan` | `plan_id` + JSON plan body | Typically a `hide_protocol::Plan` encoding |
| `evidence_graph` | nodes + typed edges | supports / refutes / cites / derived_from / same_as |
| `belief` | proposition + polarity + optional confidence | Links to evidence node ids |
| `tool_result` | tool_name, call_id, ok, JSON result | Perception `document.*` results land here |
| `bundle` | list of the above | Multi-object hop |

### Envelope

`StructuredState` adds `session_id`, `sender`, `content_hash` over the payload,
and schema `hcli.comms.structured.v0`.

### Relationship to perception

`hawking-perception::StructuredEvidence` and `Citation` should project into
`EvidenceGraph` nodes (`claim` / `evidence` / `citation`) with content hashes —
LEVEL 2 is the portable form agents exchange; perception is the producer.

---

## 6. LEVEL 3 — LATENT (experimental sealed format)

### Bible requirements (every packet)

```text
sender identity
model and revision
layer ranges
shape/dtype
visible commitment
payload hash
session
expiry
capability scope
```

### Scaffold types

- `LatentPacket` + `SealHeader`  
- `ModelIdentity` (model_id + revision + optional family)  
- `LayerRange`, `LatentDType`, `LatentKind` (hidden_state / embedding / kv_cache / other)  
- `LatentPayloadRef` — **blob pointer + size**, not live GPU tensors  
- `VisibleCommitment` — non-empty summary (auditable even if payload opaque)  
- `CapabilityScope` — default `same_model_only=true`, deny `unsealed_kv` / `cross_model_latent`  

### Policy encoded in validation

| Rule | Behaviour |
|------|-----------|
| Missing seal field | `UnsealedLatent` — refuse |
| `experimental != true` or wrong gate id | `LatentGateClosed` — refuse |
| `same_model_only` and model/revision differ | `CrossModelLatent` — refuse |
| Expiry passed | `Expired` — refuse |
| Payload bytes present but hash mismatch | `HashMismatch` — refuse |
| Acceptance after seal OK | Still returns `Deferred` — **no live transfer** |

### Bootstrap path (when later enabled)

```text
same-model Qwen 30B session
→ another Qwen 30B session
```

Helper: `LatentPacket::sealed_same_model_qwen(...)` builds a correctly sealed
header for that path without performing transfer.

### Explicit non-implementation

- No KV tensor export from Metal runtime  
- No hidden-state slice capture  
- No cross-session rebind  
- No unsealed byte pipes  

These remain experimental pending: quality improvement, latency improvement,
security gates, and auditable visible commitments (bible §20 exit conditions).

---

## 7. Outer envelope

`BusEnvelope` tagged enum:

- `text` / `structured_state` / `latent`  
- Stable `PacketId` derived from schema + level + body hash  
- `validate()` always requires full L3 seal for latent  

Routers can drop experimental traffic by level without parsing tensors.

---

## 8. Phased delivery

### Phase C0 — Scaffold (this work)

- [x] Crate `hawking-comms` with L1 / L2 / L3 types  
- [x] Seal validation + same-model policy + deferred transfer  
- [x] Unit tests for seal, cross-model refuse, hash mismatch, envelope round-trip  
- [x] This plan document  

### Phase C1 — Product integration (post–Agent OS activation)

- [ ] Map bus packets ↔ durable `hide_core::Event` kinds  
- [ ] Optional `hcli_bridge` methods or capability area documenting bus support  
- [ ] Session binding: use existing `SessionRequest` session ids only  
- [ ] Tool-result path: perception tools → L2 `ToolResult` → agent threads  

### Phase C2 — Same-model L3 experiments (owner-gated)

- [ ] Open `hcli.comms.latent.experimental` under controller policy only  
- [ ] Same Qwen revision pair only  
- [ ] Measure quality / latency / security; keep visible commitments mandatory  
- [ ] Independent evaluation — no self-promotion of latent quality  

### Phase C3 — Cross-model (blocked until evidence)

- [ ] Trained alignment artifacts  
- [ ] Independent evidence receipts  
- [ ] Explicit capability scope grants (default still deny)  

---

## 9. Security notes

- No unsealed latent/KV transfer (hard refuse).  
- Capability deny list includes `unsealed_kv`, `cross_model_latent`, `credential_read`.  
- Visible commitments must remain human-auditable.  
- Payload hash is blake3; empty or non-`blake3:` hashes unseal the packet.  
- Do not log raw latent bytes in plain product logs by default (future redaction).  

---

## 10. Non-goals

- Implementing actual KV/hidden-state transfer  
- Cross-model latent without alignment evidence  
- Replacing `hcli_bridge` control plane  
- Conflating `system_kv_bank` (serve hot path) with the agent communication bus  
- Frankenstein / live GLM recapture involvement  

---

## 11. Acceptance for C0

| Check | Evidence |
|-------|----------|
| Three levels typed | `CommLevel::{Text, StructuredState, Latent}` |
| L3 seal fields complete | `SealHeader::validate_fields` |
| Unsealed refused | missing sender → `UnsealedLatent` |
| Cross-model refused | recipient model differ → `CrossModelLatent` |
| Transfer not claimed | `validate_for_acceptance` → `Deferred` |
| Tests pass | `cargo test -p hawking-comms` |

---

## 12. Open questions (do not block C0)

1. Should L3 payload storage reuse `hide-state` capsule stores or a dedicated CAS?  
2. Event kind names for bus hops (`comms.text`, `comms.structured`, `comms.latent`)?  
3. Maximum latent payload size and residency mode interaction (§25–§27)?  
