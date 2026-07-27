# Acceleration archaeology — speculative / parallel-token zone

**Method:** same as HIDE archaeology — entry-points-inward static tracing. No
benchmarks, no edits to `tools/`, no `.gravity` artifacts, no Application Support.

| field | value |
|---|---|
| commit | `3d154422` (tree at audit; module sources last touched ~2026-07-17) |
| crate | `crates/hawking-speculate` — **6,312 LOC / 18 modules + lib.rs** |
| evidence level | STATIC_SOURCE_READING + committed docs/receipts |
| not evidence of | runtime tps, current gate pass/fail on this box, TQ parity |
| write_note | workspace root was OS write-blocked (Operation not permitted); this copy also lives under `/tmp/hawking-acceleration-archaeology/` |

**Standing thesis:** the acceleration zone is **over-built and under-wired**, with one
foundational correctness story that was *diagnosed as FAIL then fixed as a B=1 kernel
routing bug* — but campaign ledgers still record the failure as OPEN, and the TQ-served
path has never re-proven the gate. Prefer one accurate `BLOCKED` over five `REAL_WIRED`
guesses.


---

## Verdict taxonomy (HIDE method)

| verdict | meaning |
|---|---|
| `REAL_WIRED` | live entry point reaches production code that does real work |
| `REAL_UNWIRED` | real implementation, no live dispatch |
| `PARTIAL` | some live path, incomplete vs claim |
| `STUB` | scaffold / zero-token / refuse-by-construction |
| `OBSOLETE` | superseded experiment still named in ABI |
| `BLOCKED` | cannot be admitted while a named blocker holds |
| `MISSING` | plan item with no code body |


---

## A. Reachability — 18 modules

Live entry points that *can* touch speculation (all default-off except bare greedy):

| entry | path | role |
|---|---|---|
| `hawking` CLI | `crates/hawking/src/main.rs` | `--speculate`, `--user-draft`, `--eagle5-head`, `spec-oracle` subcommand |
| `hawking-serve` | `crates/hawking-serve/src/lib.rs:494-512` | `opts.speculate` → `EngineConfig` |
| Qwen generate | `crates/hawking-core/src/model/qwen_dense.rs` | user-draft / EH / eagle5 loops |
| DeepSeek-V2 generate | `crates/hawking-core/src/model/deepseek_v2.rs` | ExactShared + Eagle5 loops |
| integration tests | `crates/hawking-core/tests/*` | parity / smoke (often skip without GGUF) |

A Cargo dep of `hawking-core` on `hawking-speculate` is **not** reachability.

### Per-module table

| # | module | LOC | verdict | live entry path | missing link | confidence |
|---|---|---|---|---|---|---|
| 1 | `user_ngram` | 344 | **PARTIAL** | CLI `--user-draft` / `HAWKING_QWEN_USER_DRAFT=1` → `main.rs:3009-3010` → `qwen_dense.rs:1763-1766` → `UserNgramDraft` `:2410`/`:2643` or EH `NgramProposer` `:2675` | default OFF | high |
| 2 | `suffix_array` | 212 | **PARTIAL** | only under `HAWKING_QWEN_EVENT_HORIZON` → `qwen_dense.rs:2674-2690` `add_free_slot(SuffixArray)` + propose `:2768` | EH default OFF | high |
| 3 | `governor` | 418 | **PARTIAL** | `HAWKING_QWEN_SPEC_GOVERNOR=1` + user-draft → `qwen_dense.rs:1786,2420,2654`; also inside router | default OFF | high |
| 4 | `router` | 524 | **PARTIAL** | EH-ON only `qwen_dense.rs:2677-2835` | placeholder `target_ns_per_token=1e6` (`:2746`) | high |
| 5 | `verifier` | 195 | **PARTIAL** | EH-ON `Verifier::verify_line` `qwen_dense.rs:2821-2823`; `ExactTarget for QwenDense` `:9766` | EH-OFF uses inline verify; DeepSeek no ExactTarget | high |
| 6 | `proposal` | 150 | **PARTIAL** | trait used by EH proposers + router (`qwen_dense.rs:2673+`) | only when EH ON | high |
| 7 | `shared` | 136 | **REAL_WIRED** (opt-in) | DeepSeek ExactShared `deepseek_v2.rs:1258`; verifier; CLI `--speculate` | default `SpeculateMode::Off` | high |
| 8 | `eagle5` | 1075 | **PARTIAL / BLOCKED economics** | `--speculate eagle5` → Qwen/DeepSeek load+generate | kill-ledger NO-GO τ=0.877 | high |
| 9 | `eagle5_forward` | 499 | **PARTIAL** | Qwen matmul `qwen_dense.rs:9377,9627` | only with eagle5 | high |
| 10 | `safetensors_io` | 231 | **PARTIAL** | via `Eagle5Head::load_from_safetensors` | no path → mock | high |
| 11 | `replay_oracle` | 420 | **REAL_WIRED (tooling)** | `hawking` `main.rs:4227,4334` `replay_grid` CPU-only | not on serve/decode | high |
| 12 | `retrieval` | 317 | **REAL_UNWIRED** | **NONE FOUND** production | register + warm in EH loop | high |
| 13 | `suffix_automaton` | 486 | **REAL_UNWIRED** | **NONE FOUND** (self-gate `HAWKING_EH_SAM` only) | construct + `add_free_slot` | high |
| 14 | `parallel_draft` | 208 | **STUB** | **NONE FOUND**; emits zeros | real head + GO | high |
| 15 | `eagle_proposer` | 195 | **STUB / BLOCKED** | **NONE FOUND**; `enable_neural_slot` refuses ≠GO | τ≥2.5 oracle | high |
| 16 | `tree` | 410 | **STUB** | **NONE FOUND**; `supports_tree_verify()=false` | Metal tree verify | high |
| 17 | `cross_tokenizer` | 195 | **STUB** | **NONE FOUND** | oracle GO + text bridge | high |
| 18 | `policy` | 251 | **REAL_UNWIRED** | in router but `plan_bandit()` unused by production | call bandit or delete | high |

### Call paths (when flags set)

**User n-gram:** `hawking generate --user-draft` → `main.rs:3009` → `qwen_dense.rs:1763` → `UserNgramDraft` + `'ud_loop`.

**Event Horizon:** `USER_DRAFT=1` + `EVENT_HORIZON=1` → `qwen_dense.rs:2674` → Ngram+SuffixArray+Router → `Verifier::verify_line` `:2822`.

**Eagle5/ExactShared:** `--speculate eagle5|exact-shared` → `SpeculateMode::from_cli` → Qwen eagle loop or DeepSeek ExactShared/Eagle5.

**NONE FOUND production modules:** retrieval, suffix_automaton, parallel_draft, eagle_proposer, tree, cross_tokenizer.


---

## B. The losslessness gate — the crux

| name | where |
|---|---|
| **P0.6** | `verifier.rs:171`; EH wiring `qwen_dense.rs:2817` |
| **event_horizon_parity_prop** | `crates/hawking-core/tests/event_horizon_parity_prop.rs` |
| **user_draft_parity_e2e** | weaker single-prompt gate |
| **bsize_verify_diag** | `qwen_dense.rs` `#[ignore]` real-model matrix |
| **SPINE-5** | `M1ULTRA_RUN_REPORT.md:87` |

**Exists:** yes. **CI:** likely not (integration tests skipped / weights absent → silent skip). **Cap:** max_new_tokens=16. **Compare:** exact `Vec<u32>` vs no-spec greedy (`USER_DRAFT=0`). **Tolerance:** 0. Forces `PAIR_2R_INLINE=0` (E3 inconsistent with batched verify; production also auto-disables E3 when user-draft on — `qwen_dense.rs:5442-5445`).

### History

1. **2026-06-21 diagnosis** (`docs/plans/eh_verify_kernel_losslessness_2026_06_21.md`): **FAILED 6/20**, commit HELD.
2. **Same-day fix** `e8b92007`: root cause **`forward_tokens_verify` B==1 returned INPUT token** and mis-wrote KV — B≥2 already == greedy (margins 3.5–5.6, not near-ties). Fix: route `b==1` through `forward_token_greedy_tcb` (`qwen_dense.rs:9661-9672`). Claimed **20/20**.
3. Fix still in tree. **M1ULTRA ledger still OPEN/FAILED 6/20** — stale vs fix claim unless re-failed (no re-fail receipt found).

### Fundamental vs bug?

| claim | verdict |
|---|---|
| Original 6/20 | **IMPLEMENTATION BUG** (B==1), not proven FP non-associativity |
| B≥2 Q4_K | claimed exact; ignored regression test asserts `div_bge2==0` |
| TQ parity | **UNPROVEN / BLOCKED** (studio readiness + reentry P0) |
| This audit re-ran gate | **No** |

**Honest answer:** do **not** re-scope the tournament off bit-identity solely because of old 6/20; re-receipt Q4 and seal TQ. Do **not** treat bit-identity as proven for TQ.


---

## C. Campaign requirements vs exists

| requirement | verdict | missing link |
|---|---|---|
| EAGLE-family feature drafting | **BLOCKED** (+ code PARTIAL) | τ≥2.5 oracle |
| Native MTP | **MISSING** | native MTP head |
| Trained parallel-token heads | **STUB** | real head + GO |
| Self-speculative early exits | **MISSING** | exit head |
| Tree / multi-candidate verification | **STUB** | Metal tree verify |
| N-gram and suffix lookup | **PARTIAL** | measured costs; policy |
| Prompt/prefix lookup | **PARTIAL** | register retrieval |
| Prefix/KV/state caching | **PARTIAL** (`stateful/prefix_cache`) | draft-side use |
| Profile-specific drafting | **PARTIAL** | inject cost curve |
| Fabric-assisted draft/verify | **MISSING** | fabric→speculate ABI |


---

## D. Prior sealed negatives (binding)

| # | negative | verdict | where |
|---|---|---|---|
| 1 | EAGLE-3 / Eagle5 v3 trained head | NO-GO τ=0.877; 0.40×/0.30×/0.21× tps | `docs/dead_levers.md` |
| 2 | Eagle5 v1 routing-mask | killed (uniform MoE) | `docs/dead_levers.md` |
| 3 | Generic free proposers as default | NO-GO (τ~1.04–1.42; market often net-negative) | studio readiness |
| 4 | High accept ⇒ speedup | falsified (87% → 0.91×) | studio readiness |
| 5 | n-gram economics floor | τ~1.43 < 1.6 | M1ULTRA_RUN_REPORT |
| 6 | DeepSeek ExactShared batched full-MoE verify | reverted −17/−20 tps | `deepseek_v2.rs:1242-1253` |
| 7 | Serial verify without cheap draft | regression 2026-05-11 | dead_levers |
| 8 | Neural slots without oracle GO | structurally refused | eagle_proposer/router |
| 9 | Full spec_revive runner | `RUNNER_IMPLEMENTED=False` | studio readiness |
| 10 | SPINE-5 FAILED 6/20 OPEN | **stale ledger vs e8b92007** | M1ULTRA_RUN_REPORT |


---

## E. Naming — experiment names in ABI

| name | path |
|---|---|
| `SpeculateMode::Eagle5` | `engine.rs:90-131`, CLI, serve |
| `eagle5_head` / `Eagle5Head` / `HAWKING_QWEN_EAGLE5_*` | Qwen, DeepSeek, CLI |
| `ProposerId::Eagle5` | `router.rs:67` |
| `EagleProposer` | `eagle_proposer.rs` |
| `HAWKING_QWEN_EVENT_HORIZON` | `qwen_dense.rs` |
| `ProposerId::Rest` | router enum (paper name) |
| modules `eagle5.rs`, `eagle5_forward.rs` | crate paths |

Neutral ABI target (not built): `Off | SharedExperts | TrainedHead | FreeLookup` + `DraftProvider`.


---

## Counts

| verdict | modules | campaign items |
|---|---|---|
| REAL_WIRED | 2 (`shared` opt-in; `replay_oracle` tooling) | 0 |
| PARTIAL | 9 | 5 |
| REAL_UNWIRED | 3 | 0 |
| STUB | 4 | 3 |
| BLOCKED overlay | eagle5 / eagle_proposer | 1–2 |
| MISSING | 0 modules | 3 |

~6/18 modules on any opt-in decode path; default serve is plain greedy.


---

## Shortest missing wires

1. **Re-receipt bit-identity** on served artifact (Q4_K then TQ P0) — without this every ship claim is BLOCKED.
2. **Inject measured verifier/draft timing** into Qwen EH loop (replace placeholder 1ms target).
3. **Register RetrievalProposer + SuffixAutomaton** under EH like suffix_array.
4. **Keep neural/parallel/tree refuse-closed** until τ + 1.10 LCB oracles.
5. **Rename off Eagle5** only after 1–2.


---

## What the tournament should measure

**Admission:** (1) exact tokens vs same-target greedy; (2) accepted tok/s with full cost charging, LCB≥1.10 per workload class; (3) no sealed-kill revival without named resurrection.

**Phases (reentry P0–P6):** P0 parity → P1 cost curve → P2 free proposers → P3 AR control → P4 trained/parallel/MTP → P5 tree → P6 compose.

**Non-goals:** retrain EAGLE without pre-oracle; speedup from accept-rate alone; default-on EH before P0+costs; abandon bit-identity solely due to stale 6/20 row.

---

## Bottom line

The 6,312-line pack is a **proposal market + verify primitive**: largely implemented, largely default-off, blocked more by **sealed economics** and **unsealed TQ parity** than by missing modules. The 6/20 losslessness fail was a **B=1 routing bug fixed in `e8b92007` (claimed 20/20)**; campaign scoreboard still says OPEN; TQ never sealed.

**Program:** re-seal P0 → wire measured costs → free-proposer tournament under 1.10 LCB → only then trained/parallel/tree. One accurate `BLOCKED` beats five `REAL_WIRED` guesses.
