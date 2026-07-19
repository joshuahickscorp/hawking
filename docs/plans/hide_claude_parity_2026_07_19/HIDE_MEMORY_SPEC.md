# HIDE Memory Spec

Run date: 2026-07-19
Grounding: `HIDE_LIVE_ARCHAEOLOGY.md` (§3.5, §4, §5, config parity evidence); `HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json` (ids `config.claude_md`, `config.auto_memory`, `config.settings_precedence`); `HIDE_CLAUDE_CODE_CONFIGURATION_COMPATIBILITY.md` (§2, §3); first-pass dossier §3.2, §5.9, §5.11 "Revalidated memory" (Bible sec 45).
Status: specification for HIDE outcome-governed memory. Every load-bearing claim is tagged by the readiness of the primitive it rests on. Readiness key: **real-and-wired** / **real-but-unwired** (packed, tested, no caller) / **partial** / **stub** / **missing**.

Scope split, enforced throughout:
- **PARITY**: read an existing Claude Code project's memory and instruction tree verbatim, and record durable facts with provenance and revalidation the way Claude Code and GitHub Copilot Memory do.
- **SUPREMACY**: memory as a warm-state capsule folded into the local state fork, with no truncation cliff and deterministic offline merge. Every supremacy claim is gated on a named build item.

Memory is one of the seven things both surfaces share, never duplicated (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §1: "one memory system"). This spec defines that single system.

## 1. Thesis: memory is a governed record, not a scratchpad

Claude Code already ships two kinds of durable memory: user-curated instructions (the CLAUDE.md family) and Claude-authored auto-memory (`~/.claude/projects/<key>/memory/MEMORY.md`, first 200 lines / 25KB loaded), recalled next session [DOCUMENTED, `config.claude_md`, `config.auto_memory`]. GitHub Copilot Memory adds the governance discipline HIDE adopts: it stores repository facts **with source citations and checks those citations against the current branch before using a fact** (dossier §5.11, Bible sec 45) [DOCUMENTED].

The governance thesis: a memory record is a claim with evidence, a scope, a trust score that moves with observed outcomes, and a revalidation deadline. Successful use raises trust; guidance that leads to a failed outcome lowers trust or quarantines the record; corrections supersede rather than erase. A fact that cannot be revalidated stays available for audit but does not re-enter active context as truth. This is the opposite of a free-text notepad that silently rots and poisons.

## 2. Current readiness (verified)

| Mechanism | Readiness | Evidence |
|---|---|---|
| Memory store (SQLite / FTS5 + cosine) in `hawking-context` | **real-but-unwired** (packed @ `5a99d0e2`) | archaeology §3.5 (`hawking-context` ~4.9k, REINTEGRATE flagship) |
| Hybrid retriever (lexical + symbol + vector, RRF), BLAKE3 merkle change-gate | **real-but-unwired** (packed) | archaeology §3.5, §5 (`hawking-index` ~4.7k: FTS5+graph, PageRank repo-map, hybrid RRF, incremental MVCC daemon) |
| CLAUDE.md-tree reader (instructions) | **missing** | `config.claude_md` hide_evidence: "no CLAUDE.md reader in active tree" |
| Auto-memory reader (MEMORY.md + topic files) | **missing** | `config.auto_memory` hide_status `absent` |
| Warm-state capsule (memory-as-capsule substrate) | **partial** (RWKV atom real-but-unwired; transformer capsule missing) | `HIDE_STATE_CAPSULE_ABI.md` §2, §6; `rwkv7.rs:292-378` |
| Deterministic offline dedup / merge | **missing** (design here) | no caller in active tree |

Product status of memory today, stated plainly: the storage and retrieval primitives are real and tested but packed out of the live turn (archaeology §0, §6), and there is no reader for an existing Claude Code memory tree. Both PARITY and SUPREMACY are reintegration-plus-wiring on proven parts, not greenfield.

## 3. The memory record shape

A `MemoryRecord` is content-addressed and append-only. It is never edited in place; a change produces a new record that supersedes the old one (Section 6).

```text
MemoryRecord {
  id                // BLAKE3(claim || scope || source_set); reuses hide-security / hawking-index merkle
  claim             // the durable assertion, in the author's words
  source            // evidence set: file:line, commit sha, test id, tool receipt id, event-log id, user-turn id
  scope             // { repo_key, subtree?, user?, org?, trust_domain }   (Copilot pattern: dossier §5.11)
  confidence        // calibrated prior in [0,1], set at capture
  outcome           // governance ledger: { uses, successes, failures, last_outcome_event, trust in [0,1] }
  supersedes        // id(s) this record replaces; never a destructive rewrite
  superseded_by     // back-pointer, set when replaced
  expiry            // revalidation deadline: TTL and/or bound commit/branch the source was valid against
  retrieval_history // append-only: [ { t, query, injected: bool, outcome_link } ]
  privacy           // { sensitivity, redaction_policy, export_allowed }
  provenance        // { author: user|agent|tool, capture_boundary, injection_trust_label }
  status            // active | probation | quarantined | superseded | expired
}
```

Rules:
- **No claim without a source.** A record with an empty `source` set is captured `probation` at best and is never injected as truth (mirrors Copilot's citation requirement, dossier §5.11).
- **`id` is content-addressed**, so identical facts from two paths dedup deterministically (Section 10).
- **`retrieval_history` links to outcomes**, not just queries, so trust in Section 4 is computed from evidence, not from how often a record was merely fetched.

## 4. Outcome governance (trust dynamics)

Every record carries a `status` in a fixed lifecycle and a `trust` scalar that moves with observed outcomes.

```text
capture ─▶ probation ─▶ active ─▶ (superseded | expired)
              │            │
              └──────▶ quarantined ◀──────┘
```

| Transition | Trigger | Rule |
|---|---|---|
| capture → probation | new record with sources but no outcomes yet | usable as a hint, labeled unproven; not injected as asserted truth |
| probation → active | first successful use, or deterministic corroboration | trust raised |
| active → active (trust up) | a turn that used the record reaches a passing deterministic oracle (test/build/policy) or a graded acceptance | `outcome.successes++`, trust raised |
| active → probation/quarantined | a turn that used the record reaches a failing oracle attributable to the record | `outcome.failures++`, trust lowered; below a floor, quarantine |
| any → quarantined | revalidation fails (Section 5), or injection-provenance flag set (Section 8) | removed from active context, retained for audit |
| any → superseded | a correction supersedes it (Section 6) | retained, back-pointer set |
| any → expired | `expiry` passed and revalidation not renewed | retained for audit, not injected |

Governance discipline (dossier §5.8 actor/evaluator separation, applied to memory):
- **Deterministic oracles first.** Trust rises or falls on test/build/policy receipts where one exists. A prose model-evaluator may adjust trust only for claims with no deterministic oracle, and **may never overrule a failing deterministic gate** to keep a record active.
- **Attribution before penalty.** A failure lowers a record's trust only when the record was actually injected into the failing turn (`retrieval_history.injected == true`) and is plausibly implicated; unrelated failures do not punish it.
- **Quarantine is reversible and audited.** A quarantined record is never deleted; it can be restored by an explicit user action or by later corroboration. This is the audit-not-erase posture (dossier §5.11: "stale or unverified facts remain available for audit but do not enter active context as truth").

## 5. Revalidation against the current branch (Copilot pattern)

Before a record is injected into active context, its sources are revalidated against the **current** repository state. [DOCUMENTED pattern, dossier §5.11]

- **Citation check.** For each `source` that references code (file:line, symbol, commit), confirm it still resolves and still supports the claim. Use `hawking-index`'s BLAKE3 merkle change-gate (archaeology §5, `hawking-index` change-gate) to answer "did anything the sources depend on change since capture" in O(changed subtree), not a full re-scan.
- **Outcomes on stale sources.** If a source no longer resolves or the merkle gate reports the cited region changed, the record does not enter context as truth: it drops to `probation`/`quarantined` and is queued for re-derivation, while remaining queryable for audit.
- **Branch binding.** `expiry` may bind a record to a commit or branch its evidence was valid against; a branch switch triggers revalidation rather than blind reuse. This directly answers the injection-persistence risk in Section 8 (a poisoned fact cannot ride a branch switch into a fresh context unchecked).

Revalidation is a read against the live index; it does not require a model call, so it is cheap enough to run at every injection.

## 6. Supersession, not destructive rewrite

Corrections **supersede**; they never overwrite (dossier §3.2 "durable memory records with provenance and supersession"; §5.11 "corrections should supersede rather than erase history"). This mirrors the state-capsule discipline where `fork` is copy-not-merge and there is deliberately no blend inverse (`HIDE_STATE_CAPSULE_ABI.md` §7).

- A correction is a new `MemoryRecord` whose `supersedes` names the prior id; the prior record gets `superseded_by` and `status = superseded`, and is retained.
- Active retrieval returns only the head of a supersession chain; audit and time-travel can walk the full chain.
- Supersession preserves the outcome ledger of the chain, so a fact that was corrected once and is failing again does not re-enter with a laundered trust score.

There is no in-place mutation API. This is what makes memory replayable and lets the event log (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §2, durable truth plane) reconstruct memory state at any point.

## 7. Hybrid retrieval

Retrieval fuses three signals over `hawking-index` and the `hawking-context` memory store, both real-but-unwired (packed @ `5a99d0e2`, archaeology §3.5):

| Signal | Source | Answers |
|---|---|---|
| Lexical | FTS5 full-text (`hawking-index`, `hawking-context` memory store) | exact tokens, error strings, identifiers |
| Symbol | tree-sitter defs/refs + SCIP ids + PageRank repo-map (`hawking-index`) | "facts about this symbol / this file / its callers" |
| Vector | cosine over embeddings (`hawking-context` memory store) | paraphrastic / semantic recall |

Fusion is **reciprocal-rank fusion (RRF)**, the hybrid retriever already built in `hawking-index` (archaeology §5). Retrieval is always **scope-filtered first** (`repo_key`, subtree, user, org, trust_domain: dossier §5.11 "scope memory by repository, user, organization, and trust domain") and **status-filtered** (only `active` head records enter context as truth; `probation` may enter labeled as a hint; `quarantined`/`superseded`/`expired` are audit-only). Injected records append to their `retrieval_history` so Section 4 can attribute outcomes.

## 8. Memory as an injection-persistence surface (security)

Persistent memory is an attack surface: the dossier names "persistent-memory poisoning" among the concrete threats (§5.9) and requires "memory records treated as an injection persistence surface" (§5.9). The danger is a hostile string, arriving from tool output, a fetched page, or an untrusted config file parsed before a trust decision, that gets written as a durable fact and then silently steers every future session.

HIDE defenses (this system implements them; enforcement primitives live in `hide-security`, see `HIDE_SECURITY_CONSTITUTION.md`):

- **Immutable provenance label on capture.** Every record records `provenance.author` (user | agent | tool) and `provenance.injection_trust_label`. Content authored from untrusted material inherits an untrusted label that cannot be edited off the record.
- **Gate G-MEM-1 (revalidate before truth):** no record enters active context as asserted truth without passing the Section 5 citation check against the current commit. A fact with no live, resolving evidence is a hint at most.
- **Gate G-MEM-2 (untrusted capture is quarantined):** a record whose author is `tool` or whose source is untrusted external content (web fetch, imported config before the folder trust boundary, dossier §5.9) is captured `quarantined`. It cannot be auto-promoted to `active`; promotion requires an explicit user action **or** independent deterministic corroboration (a passing oracle that does not itself depend on the untrusted record).
- **Sanitization before context.** Untrusted content is inspected/sanitized before it can enter model context (dossier §5.9 "inspection/sanitization before untrusted output enters model context"); memory writes derived from it are subject to the same boundary.
- **Scope containment.** Records do not cross `trust_domain` on retrieval (Section 7), matching the capsule rule that state does not cross security domains (`HIDE_STATE_CAPSULE_ABI.md` §4). A workspace cannot read another workspace's memory.
- **Audit export without leaking secrets.** `privacy.export_allowed` and redaction policy gate audit export, honoring the dossier's "audit export without exposing sensitive content by default" (§5.9).

Trust-before-config ordering is a hard prerequisite: memory writes from project-local material are gated behind the folder trust boundary, never captured before it (dossier §5.9; `HIDE_CLAUDE_CODE_CONFIGURATION_COMPATIBILITY.md` §7 item 5, "never before trust").

## 9. Migration layer (PARITY): read an existing project verbatim

On pointing HIDE at an existing Claude Code repository, the migration reader ingests both durable-memory classes verbatim, so a project migrates with minimal rewriting (Bible §18 goal). Exact locations, precedence, and reader obligations are specified in `HIDE_CLAUDE_CODE_CONFIGURATION_COMPATIBILITY.md`; this spec references them rather than restating them.

| Migration source | Ingest as | Reference |
|---|---|---|
| CLAUDE.md tree (`./CLAUDE.md`, `./.claude/CLAUDE.md`, ancestors root-first, subdir lazy-on-touch, `CLAUDE.local.md`) | user-authored `MemoryRecord`s, scope by directory subtree; author = user; high initial confidence | CONFIG_COMPATIBILITY §2; `config.claude_md` |
| `@path` imports (depth 4, code-spans skipped, first-use approval) | resolved and folded into the importing record's source set; import boundary preserved | CONFIG_COMPATIBILITY §2 |
| `.claude/rules/**/*.md` (un-scoped load at launch; `paths:`-globbed rules load on matching Read) | scoped `MemoryRecord`s carrying the rule's path predicate as scope | CONFIG_COMPATIBILITY §2 |
| Auto-memory `~/.claude/projects/<repo-key>/memory/MEMORY.md` (200 lines / 25KB) + linked topic files | Claude-authored `MemoryRecord`s; author = agent; captured `probation` pending first revalidation | CONFIG_COMPATIBILITY §3; `config.auto_memory` |

Reader obligations inherited from CONFIG_COMPATIBILITY (do not re-implement here): the two independent precedence orders (instructions `Managed>User>Project>Local` read-last-wins; settings `Managed>CLI>Local>Project>User` with permissions merging), HTML-comment stripping, `claudeMdExcludes`, and re-reading root CLAUDE.md after compaction.

**Quick-capture affordance.** Claude Code's inline capture is commonly described with a `#` shortcut, but the `#` name is **not in current Claude docs** [verifier, `config.auto_memory`, CONFIG_COMPATIBILITY §3, §8]. HIDE therefore implements the **capability** (a one-keystroke "remember this" that writes a `MemoryRecord` with the current turn as source and user as author), not the named feature; it does not name-copy `#`.

Honest boundary: migration guarantees reading these files with the right semantics and precedence, not bug-for-bug behavior of arbitrary hook/skill scripts (CONFIG_COMPATIBILITY §8). Read fidelity is a test target (CONFIG_COMPATIBILITY §7).

## 10. Hawking superiority (SUPREMACY): memory as a warm-state capsule

Claude Code re-sends its memory as text every session and truncates auto-memory at 200 lines / 25KB. Because HIDE runs the model locally, the resolved memory set can be compiled once and folded into the resident execution state, so it is free after the first fork rather than re-tokenized each turn. This is the same "pass state, not text" moat as `HIDE_STATE_CAPSULE_ABI.md`, applied to memory.

| Supremacy claim | Structural basis | Gated on |
|---|---|---|
| No 200-line / 25KB truncation cliff | resolved memory compiled into a warm prefix/state capsule, not a fixed prompt budget | context compiler wired (`HIDE_CONTEXT_OS_SPEC.md`) + capsule exposure (`HIDE_STATE_CAPSULE_ABI.md` §8) |
| No re-tokenization each session | injected memory rides the fork as compute state; a surface switch is a pointer copy | RWKV lane: fork is real-but-unwired (`rwkv7.rs:376-378`); transformer lane: capsule **missing** (KvCache not serializable) |
| Deterministic offline dedup / merge | content-addressed `id` (Section 3) makes identical facts collapse; supersession chains merge by rule, offline, no model call | this spec + `hawking-context` store wired |
| Memory revalidation with no round-trip | citation check runs against a local index (Section 5) | `hawking-index` wired (packed today) |

**Gate G-MEM-3 (capsule supremacy is gated):** until the state-capsule exposure build items land (`HIDE_STATE_CAPSULE_ABI.md` §8: serve state routes, session→slot affinity, `SstateDiskCache` persistence) and the context compiler is wired (`HIDE_CONTEXT_OS_SPEC.md`), HIDE memory is a durable **record store plus hybrid retrieval plus revalidation**, and injected memory is re-tokenized each session exactly like Claude Code. The truncation-cliff and re-tokenization wins are real and structural, but they are not shipping today. The `config.claude_md` and `config.auto_memory` superiority notes ("compile into a warm-state capsule / prefix KV") are claims **gated on these build items**, not current behavior.

Capsule caveat inherited from the ABI: the transformer / Hybrid capsule (needed for Qwen3-Coder-Next-class models) is unbuilt, so memory-as-capsule ships first on the RWKV lane; the durable record store, governance, revalidation, and migration (Sections 3 to 9) are architecture-independent and can ship on any model once reintegrated.

## 11. What this buys, stated conservatively

| Claim | Status | Basis |
|---|---|---|
| Read an existing project's CLAUDE.md tree + imports + rules + auto-memory verbatim | build item (reader **missing**), no invention needed | `config.claude_md`/`config.auto_memory` absent; CONFIG_COMPATIBILITY specifies the target |
| Record durable facts with provenance, citations, supersession | supported once reintegrated | `hawking-context` store + `hawking-index` retriever real-but-unwired (§2) |
| Revalidate a fact against the current branch before use | supported once reintegrated | merkle change-gate real (archaeology §5); Copilot pattern (dossier §5.11) |
| Trust that rises with passing oracles, falls / quarantines on failure | supported once reintegrated | needs the flat kernel loop + eval harness wired (archaeology §6) |
| Resist persistent-memory poisoning (quarantine untrusted, scope-contain) | supported; enforcement primitives in `hide-security` real-but-unwired | dossier §5.9; `HIDE_SECURITY_CONSTITUTION.md` |
| Memory with no truncation cliff, folded into the fork, free after first use | **not yet** | gated by G-MEM-3 (capsule exposure + compiler); RWKV lane only at first |
| Deterministic offline dedup / merge | supported once store wired | content-addressed ids (§3), no model call |

Every parity item is a reader or a reintegration of a proven part. Every supremacy item is gated on a named build item, never asserted as shipping.

## Gate summary

| Gate | Statement |
|---|---|
| G-MEM-1 | No record enters active context as truth without passing revalidation (citation check vs current commit). |
| G-MEM-2 | Records authored from untrusted content are captured `quarantined` and cannot auto-promote; promotion needs an explicit user action or independent deterministic corroboration. |
| G-MEM-3 | Warm-state-capsule memory (no truncation cliff, no re-tokenization) is gated on state-capsule exposure (`HIDE_STATE_CAPSULE_ABI.md` §8) and a wired context compiler (`HIDE_CONTEXT_OS_SPEC.md`); until then memory is a record store plus retrieval plus revalidation, re-tokenized each session. |
