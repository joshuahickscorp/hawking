# Acceleration archaeology — speculative / parallel-token zone

**Method:** entry-points-inward static tracing (HIDE method). No benches, no `tools/` edits, no `.gravity` artifacts.

| field | value |
|---|---|
| commit (audit tree) | `3d154422` (module sources ~2026-07-17) |
| crate | `crates/hawking-speculate` — **6,312 LOC / 18 modules + lib.rs** |
| evidence | STATIC_SOURCE_READING + committed docs/receipts |
| not evidence of | runtime tps, live gate pass/fail, TQ parity |

**Thesis:** acceleration zone is **over-built and under-wired**. Foundational losslessness fail was a **B=1 kernel routing bug fixed** (`e8b92007`, claimed 20/20), but campaign ledgers may still show OPEN and TQ-served path never re-proved the gate. Prefer one accurate `BLOCKED` over five `REAL_WIRED` guesses. Historical only for constitution **BC-ACCEL-009** (Event Horizon / EAGLE5 product-released absent); live env/CLI controls are not product capability — see `docs/env_flags.md` Spec-decode and CAP released row.

## Verdict taxonomy

| verdict | meaning |
|---|---|
| `REAL_WIRED` | live entry reaches production work |
| `REAL_UNWIRED` | real impl, no live dispatch |
| `PARTIAL` | some live path, incomplete |
| `STUB` | scaffold / refuse-by-construction |
| `BLOCKED` | named blocker holds |
| `MISSING` | plan item, no body |

## Reachability

Live entries (default-off except bare greedy): `hawking` CLI (`--speculate`, `--user-draft`, `--eagle5-head`, `spec-oracle`); `hawking-serve` `opts.speculate`; Qwen/DeepSeek generate loops; integration tests (often skip without GGUF). Cargo dep alone ≠ reachability.

| # | module | LOC | verdict | live path / gap |
|---|---|--:|---|---|
| 1 | `user_ngram` | 344 | PARTIAL | `--user-draft` / `HAWKING_QWEN_USER_DRAFT`; default OFF |
| 2 | `suffix_array` | 212 | PARTIAL | EH only; default OFF |
| 3 | `governor` | 418 | PARTIAL | `HAWKING_QWEN_SPEC_GOVERNOR` + user-draft |
| 4 | `router` | 524 | PARTIAL | EH-ON; placeholder `target_ns_per_token=1e6` |
| 5 | `verifier` | 195 | PARTIAL | EH `Verifier::verify_line`; DeepSeek lacks ExactTarget |
| 6 | `proposal` | 150 | PARTIAL | EH proposers + router |
| 7 | `shared` | 136 | REAL_WIRED opt-in | DeepSeek ExactShared; default Off |
| 8 | `eagle5` | 1075 | PARTIAL / BLOCKED econ | `--speculate eagle5`; kill-ledger NO-GO τ=0.877 |
| 9 | `eagle5_forward` | 499 | PARTIAL | only with eagle5 |
| 10 | `safetensors_io` | 231 | PARTIAL | via Eagle5Head load |
| 11 | `replay_oracle` | 420 | REAL_WIRED tooling | CLI `replay_grid` CPU-only |
| 12 | `retrieval` | 317 | REAL_UNWIRED | none production |
| 13 | `suffix_automaton` | 486 | REAL_UNWIRED | self-gate only |
| 14 | `parallel_draft` | 208 | STUB | zeros zeros |
| 15 | `eagle_proposer` | 195 | STUB / BLOCKED | refuses ≠GO; τ≥2.5 |
| 16 | `tree` | 410 | STUB | `supports_tree_verify()=false` |
| 17 | `cross_tokenizer` | 195 | STUB | none |
| 18 | `policy` | 251 | REAL_UNWIRED | `plan_bandit()` unused |

**Call paths (flags on):** user n-gram → `UserNgramDraft`; Event Horizon → Ngram+SuffixArray+Router+Verifier; Eagle5/ExactShared via `SpeculateMode`. Production NONE: retrieval, suffix_automaton, parallel_draft, eagle_proposer, tree, cross_tokenizer.

## Losslessness gate (crux)

| name | where |
|---|---|
| P0.6 | `verifier.rs`; EH wiring `qwen_dense.rs` |
| event_horizon_parity_prop | core tests |
| user_draft_parity_e2e | weaker single-prompt |
| SPINE-5 | M1ULTRA run report |

Exact `Vec<u32>` vs no-spec greedy; tolerance 0; max_new_tokens=16; forces `PAIR_2R_INLINE=0` under user-draft. History: 2026-06-21 **FAILED 6/20** (held); same-day fix `e8b92007` — B==1 `forward_tokens_verify` returned input token / mis-wrote KV; route b==1 through `forward_token_greedy_tcb`; claimed **20/20**. Fix still in tree; ledgers may be stale OPEN. **TQ parity UNPROVEN/BLOCKED.** Do not abandon bit-identity solely from stale 6/20; re-receipt Q4 then seal TQ.

## Campaign vs exists + sealed negatives

| requirement | verdict |
|---|---|
| EAGLE-family drafting | BLOCKED (+ PARTIAL code); τ≥2.5 |
| Native MTP / early exits / fabric draft | MISSING |
| Parallel-token / tree verify | STUB |
| N-gram/suffix/prefix/profile drafting | PARTIAL |

Binding negatives (see also `docs/dead_levers.md`): EAGLE-3 NO-GO τ=0.877 net-negative tps; Eagle5 v1 routing dead; free proposers τ~1.04–1.42 often net-negative; high accept ≠ speedup (87%→0.91×); n-gram τ~1.43; ExactShared batched full-MoE verify reverted; serial verify regression; neural slots refuse without oracle; `RUNNER_IMPLEMENTED=False`; SPINE-5 OPEN stale vs fix.

## Naming / counts / wires

ABI still names `SpeculateMode::Eagle5`, `Eagle5Head`, `HAWKING_QWEN_EVENT_HORIZON`, `ProposerId::Eagle5|Rest`. Neutral target (not built): `Off|SharedExperts|TrainedHead|FreeLookup` + `DraftProvider`.

| verdict | modules |
|---|--:|
| REAL_WIRED | 2 |
| PARTIAL | 9 |
| REAL_UNWIRED | 3 |
| STUB | 4 |

~6/18 modules on any opt-in decode path; default serve is plain greedy.

**Shortest wires:** (1) re-receipt bit-identity on served artifact; (2) inject measured verifier/draft costs into EH; (3) register Retrieval+SuffixAutomaton under EH; (4) keep neural/parallel/tree refuse-closed until τ+1.10 LCB; (5) rename off Eagle5 after 1–2.

**Tournament admission:** exact tokens vs greedy; accepted tok/s full cost LCB≥1.10; no sealed-kill revival without named resurrection. Phases P0 parity → P1 costs → P2 free proposers → P3 AR → P4 trained/MTP → P5 tree → P6 compose.

## Bottom line

The 6,312-line pack is a **proposal market + verify primitive**: largely implemented, largely default-off, blocked more by **sealed economics** and **unsealed TQ parity** than missing modules. Program: re-seal P0 → measured costs → free-proposer tournament under 1.10 LCB → only then trained/parallel/tree.
