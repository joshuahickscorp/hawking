# Runtime-status audit — prefix-cache (L1.2) & vocab/draft specialization (L3.1)

**Date:** 2026-05-31 · **Task E** (GPU-free; read+reason only) · **Verdict: NEEDS-MEASUREMENT** (one default-on path lacks a decode-loop parity test; see §4)

Purpose: separate what is **actually wired into the runtime** from what is
only **designed / oracle-d**. Every claim below is tagged
`(measured)` / `(proxy)` / `(estimate)` / `(code-read)` and points at the
exact source line. No GPU was run; the wiring facts are static reads of
`crates/dismantle-core/src/`, the perf numbers are quoted from prior
commits/oracles and re-tagged honestly.

---

## 1. Status table

| Lever | Module / site | Status | Controlling flag | Default | Claimed effect | Parity/bit-identity test |
|---|---|---|---|---|---|---|
| **L1.2 RAM prefix cache** (Track A) | `stateful/prefix_cache.rs` (`InMemoryPrefixCache`), wired in `model/qwen_dense.rs:1361-1407` (lookup/restore) + `:1547-1559` (store) | **WIRED + ON-BY-DEFAULT** | `DISMANTLE_QWEN_PREFIX_CACHE` via `env_opt_out` (`lib.rs:41`) — on unless set to `0/false/off/no` | **ON** | TTFT/prefill cut on a shared-prefix turn-2 prompt | `ram_prefix_cache_e2e.rs` — **TCB path only** (see §4 gap) |
| **L1.2 disk prefix cache** (Track B) | `cache/prefill_disk.rs` (`PrefillDiskCache`), wired in `qwen_dense.rs:1409-1446` + store `:1539-1545` | **WIRED + ENV-GATED** | `DISMANTLE_PREFIX_CACHE_DIR` (`prefill_disk.rs:226`); budget `DISMANTLE_PREFIX_CACHE_BUDGET_MB` | **OFF** (no dir set ⇒ `open_from_env` returns `None`) | persistence tier (survives restart); only consulted when RAM tier missed (`qwen_dense.rs:1416`) | `prefix_cache_parity.rs` (synthetic) + `prefix_cache_e2e.rs` (real Qwen) — both green per headers |
| **L3.1(a) Qwen vocab-prune** (the live one) | **inline in `qwen_dense.rs:988-1152`**; state in fields `vocab_pruned` / `vocab_prune_remap` (`:214,224`); argmax remap `:4578`, `:5280` | **WIRED + ENV-GATED** | `DISMANTLE_QWEN_VOCAB_PRUNE_CORPUS=N` (freq+remap) **or** `DISMANTLE_QWEN_VOCAB_PRUNE=N` (legacy first-N, identity remap) | **OFF** | LM-head + argmax compute cut (76.9% rows at N≈23.6K); ~+1-3 tps standalone *(estimate, from `vocab_prune.rs` header / path-to-30)* | **partial** — `user_draft_parity_e2e.rs` test #2 sets `VOCAB_PRUNE=32000`+`Q4K_LMHEAD=1` and asserts bit-identity, but only the **first-N** prune; the **CORPUS-remap** path has no dedicated test (§4) |
| **L3.1(a) `PrunedVocab` module** | `vocab_prune.rs` (`PrunedVocab::load/slice_lm_head_f16/pruned_to_original`) | **DESIGNED + WIRED FOR DEEPSEEK ONLY — DEAD CODE on the Qwen path** | `EngineConfig.vocab_prune_path` (`engine.rs:34`) → consumed **only** by `deepseek_v2.rs:500,529,1623` | n/a for Qwen | whitelist-JSON prune of V2-Lite LM head | unit tests in-module (12) + `vocab_prune_parity.rs` runs **DeepSeek-V2** weights only (`tests/vocab_prune_parity.rs:27`) |
| **L3.1(b) per-user n-gram draft** | `speculate/user_ngram.rs` (`UserNgramDraft`), decode loop in `qwen_dense.rs:2229-2390` | **WIRED + ENV-GATED** | `DISMANTLE_QWEN_USER_DRAFT` (`env_on`, `:1640`); K = `DISMANTLE_QWEN_USER_DRAFT_K` (default 4, cap 8). Requires `DISMANTLE_QWEN_TCB=1` + temp=0 + **not** eagle5 (`:1637-1640`) | **OFF** | lossless-by-construction speedup on repetitive code; τ≈3.40 warm-start *(proxy)*; +148% on repetitive code *(measured, single prompt — per MEMORY draft_tuning_verify)* | `user_draft_parity_e2e.rs` test #1 — real Qwen, asserts 16-tok `==` (draft-ON vs OFF) |
| **L1.2 semantic layer** (near-dup reuse) | `prefix_cache.rs:670-702` (`SemanticIndex` trait, `SemanticCandidate`) | **DESIGNED-ONLY** (trait, no impl, no wiring) | none | n/a | post-exact-match near-dup reuse (Q→E after verify) | none (no body) |

---

## 2. What is genuinely live by default vs opt-in

- **On by default, in every `QwenDense::generate`:** the **in-RAM prefix
  cache** only. `env_opt_out` returns `true` when the var is unset
  (`lib.rs:41-49`), and `qwen_dense.rs:1372` allocates the cache on first
  generate. It sits in front of the disk tier; the disk tier is skipped
  entirely when the RAM tier already covered the prefix
  (`qwen_dense.rs:1416`). `(code-read)`
- **Opt-in (all default-OFF):** disk prefix cache (needs
  `DISMANTLE_PREFIX_CACHE_DIR`), Qwen vocab-prune (needs
  `DISMANTLE_QWEN_VOCAB_PRUNE[_CORPUS]`), per-user n-gram draft (needs
  `DISMANTLE_QWEN_USER_DRAFT` **and** `DISMANTLE_QWEN_TCB` **and** temp=0).
  `(code-read)`
- **Designed-only / not in the forward pass:** `PrunedVocab` for Qwen (it is
  the DeepSeek-V2 prune, never called from `qwen_dense.rs`), and the
  `SemanticIndex` near-duplicate layer. `(code-read)`

## 3. The "~84% prefill, GO" claim — disambiguated (the MEMORY line conflates two numbers)

The MEMORY entry "prefix-cache ~84% prefill, GO" actually merges **two
distinct results**, and they should not be quoted as one:

1. **~84% prefill cut** = a **single-pair micro-bench** `(measured)`:
   commit `ebfc57a` recorded one turn-2 prompt sharing turn-1's prefix going
   **5551 ms → 892 ms** prefill (~84%). This is **one prompt pair on the TCB
   path**, a TTFT win (not dec_tps). It is real but n=1; it is the
   *mechanism working*, not a session-level rate.
2. **86.8% session reuse / 91.8% mean shared-prefix** = a **PROXY**
   `(proxy)`: `reports/oracle_prefix_cache.md` /
   `tools/bench/oracle_prefix_cache.py`, run on a **fabricated** 24-request
   coding session (no real served transcripts exist on disk). The report
   itself flags "THIS IS A PROXY, NOT A REAL MEASUREMENT." 4 of 23 pairs are
   context-switches in the 30-70% buckets.

So the honest statement is: *the cache is bit-identical and live by default;
a single measured pair shows ~84% prefill reduction when a prefix actually
matches; the session-level hit-rate that decides how often that happens is
still only a proxy (~87%), pending the user's real TailorAI transcripts.*
The GO verdict in `oracle_prefix_cache.md` is **a GO on the proxy**, not on
production traffic.

## 4. Specific gaps (what the GPU lane should re-confirm)

1. **DEFAULT-ON path with a coverage hole — the non-TCB prefix-restore.**
   `ram_prefix_cache_e2e.rs:64` sets `DISMANTLE_QWEN_TCB=1`, so the only
   real-model bit-identity gate for the RAM cache exercises the **GPU TCB**
   restore path. But the cache is **on by default even without TCB**
   (`forward_token`, `qwen_dense.rs:1481-1497,1499-1511`), and that path
   restores `self.kv` directly (the arena-mirror at `:1529` is a
   `#[cfg(macos)]`/TCB concern). **No e2e test asserts cache-hit == cache-off
   on the non-TCB `forward_token` decode path.** The synthetic
   `prefix_cache_parity.rs` covers the disk-tier restore *math* with a
   `fake_forward`, not the RAM tier through the real model on the CPU/hybrid
   path. → **GPU lane: run a real-Qwen hit-vs-off 16-tok `==` gate with
   `DISMANTLE_QWEN_TCB` UNSET** to close the default-on path.

2. **Stale module docstrings (high trap risk for the next session).**
   Both `stateful/mod.rs:9-12` ("# Status: INTERFACES ONLY … Every function
   body … is `todo!()`") and `prefix_cache.rs:11` ("# Status: INTERFACES
   ONLY — all bodies are `todo!()`") are **false**: `prefix_cache.rs` has
   zero `todo!()` in code (the only match is in the doc prose) and is fully
   wired + on by default. `prefix_cache.rs:27-28` also asserts "This module
   does **not** modify `qwen_dense.rs`" — true of the file, but the wiring
   that consumes it lives in `qwen_dense.rs:1361-1559`. → **Doc-only fix
   (not this task's scope): mark prefix_cache as SHIPPED/default-on.** Anyone
   trusting these headers would wrongly conclude L1.2 is unbuilt.

3. **Qwen CORPUS-remap vocab-prune is untested for bit-identity.** The only
   bit-identity assertion touching Qwen vocab-prune
   (`user_draft_parity_e2e.rs` test #2) uses the **legacy first-N** prune
   (`VOCAB_PRUNE=32000`, where pruned_idx ≡ original_id, so the remap is the
   identity and the argmax→token translation at `qwen_dense.rs:4578/5280` is
   trivially correct). The **frequency-corpus** path
   (`DISMANTLE_QWEN_VOCAB_PRUNE_CORPUS`, `qwen_dense.rs:1025-1072`) builds a
   **non-identity remap** and is the one that can actually mis-map a token —
   it has **no dedicated parity test**. → **GPU lane: add/realize a
   real-Qwen `VOCAB_PRUNE_CORPUS=N` greedy-equality gate** (a held-out prompt
   whose emitted tokens all survive the whitelist), mirroring
   `vocab_prune_parity.rs`'s logic but on the Qwen inline path.

4. **`vocab_prune.rs` is dead code on the Qwen path — verify intent.** The
   `PrunedVocab` module is referenced only by `deepseek_v2.rs`. If V2-Lite is
   no longer a shipped target, `vocab_prune.rs` + `EngineConfig.vocab_prune_path`
   + `vocab_prune_parity.rs` are maintaining a path the production model never
   takes. Not a correctness bug; a possible cleanup. *(flagging only, no
   action this task.)*

5. **Per-user draft hard-requires TCB + greedy.** `use_user_draft`
   (`qwen_dense.rs:1637`) is `false` unless `DISMANTLE_QWEN_TCB=1` **and**
   `temperature == 0.0` **and** eagle5 is off. A non-TCB or sampled run
   silently gets plain greedy (lossless, but the lever is a no-op). The
   parity test sets TCB=1 (`user_draft_parity_e2e.rs:54`); there is no test
   that the *guard* correctly disables the draft under temp>0 — low risk
   (output is the verifier's either way) but worth a one-line note.

## 5. The single decisive gate

For each lever, the one thing that would settle "is this real in production":

- **RAM prefix cache:** a real-Qwen **non-TCB** hit-vs-off 16-tok `==` gate
  (closes gap #1) — *correctness*; and the **real TailorAI session-transcript
  hit-rate** via `oracle_prefix_cache.py` Mode-1 (closes the proxy in §3) —
  *value*. The mechanism (bit-identity + ~84% on a hit) is already measured;
  what's unmeasured is **how often a hit happens on this user's real traffic.**
- **Qwen vocab-prune:** a real-Qwen `VOCAB_PRUNE_CORPUS` greedy-equality gate
  (gap #3) — without it the corpus-remap path's bit-identity is asserted only
  by the inline comments, not a test.
- **Per-user n-gram draft:** already has the lossless bit-identity gate
  (`user_draft_parity_e2e.rs`, real Qwen, 16-tok `==`); the open question is
  **paired dec_tps** on representative (not best-case-repetitive) prompts —
  current "+148%" is a single repetitive-code prompt `(measured, n=1)`.

---

### Provenance (all static reads, 2026-05-31)
- Wiring: `crates/dismantle-core/src/model/qwen_dense.rs` (lines cited inline),
  `src/lib.rs:32-49` (env helpers), `src/stateful/{mod.rs,prefix_cache.rs}`,
  `src/speculate/user_ngram.rs`, `src/vocab_prune.rs`,
  `src/cache/prefill_disk.rs:225-239`, `src/engine.rs:34`,
  `src/model/deepseek_v2.rs:500-529,1623`.
- Tests: `tests/{ram_prefix_cache_e2e,prefix_cache_e2e,prefix_cache_parity,user_draft_parity_e2e,vocab_prune_parity}.rs`.
- Perf claims: commit `ebfc57a` body (~84% prefill, measured n=1);
  `reports/oracle_prefix_cache.md` (86.8% reuse, proxy);
  MEMORY `draft_tuning_verify_findings` (+148%, measured n=1).
- Git landing: `b2533d7` (scaffold) → `ebfc57a` (RAM cache opt-in) →
  `fc93ea0` (default-ON + bounded budget) → `cdb28f2` (UserNgramDraft).
