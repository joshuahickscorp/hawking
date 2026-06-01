# MOVE-2 — user-ngram-draft root cause (read-only code synthesis, 2026-05-31)

Read-only. No `cargo test`, no `generate`, no GPU. Conclusions are from code
logic against `crates/dismantle-core/src/model/qwen_dense.rs`,
`crates/dismantle-core/src/speculate/user_ngram.rs`,
`crates/dismantle-core/tests/user_draft_parity_e2e.rs`,
`tools/bench/clean_room_batch.sh`, and the prior session's
`plans/draft_tuning_verify_findings_2026_05_31.md` (commit `354d718`). Numbers
re-stated from that doc are tagged `(measured, contaminated)`; everything else
is `(code-logic)`.

---

## 0. The two observations, reconciled

There are **two different runs** in the handoff and they are NOT in conflict:

| run | env | result |
|---|---|---|
| parity test `user_draft_is_bit_identical` | `TCB=1`, `PREFIX_CACHE=0`, **default cfg** (no VOCAB_PRUNE/Q4K_LMHEAD), `USER_DRAFT=1` | `draft_accepted=7/16`, bit-identical (CPU fp16 fallback verify) `(measured)` |
| parity test `user_draft_bit_identical_fast_pruned_q4k` (`354d718`) | `+ VOCAB_PRUNE=32000 + Q4K_LMHEAD=1`, `USER_DRAFT=1` | `draft_accepted=7`, bit-identical (GPU fast verify) `(measured)` |
| findings-doc in-session table | full fast env (`+ FFN_DOWN_Q4K + Q4K_PREDEC`), **`USER_DRAFT=1`** | `draft_accepted=3`, 25.9 tps `(measured, contaminated)` |
| **the failing CLI run** | full fast env, **`USER_DRAFT` NOT set**, `VERIFY_TIMING` NOT set | **`draft_accepted=0`, ZERO `[verify-timing]` lines** |

The findings doc proves the draft **does** fire under the full fast env when
`USER_DRAFT=1` (it got `draft_accepted=3`, not 0). Therefore the failing CLI
run's `draft_accepted=0` is **not** a Q4K/FFN/PREDEC bug in the draft path. It
is the third row vs the fourth row: the fourth row never turned the draft on.

---

## 1. PINNED ROOT CAUSE

**The full-fast-env CLI run never entered the user-draft branch because
`DISMANTLE_QWEN_USER_DRAFT` was not set, so `use_user_draft` evaluated to
`false` and `generate()` fell through to the plain greedy-decode `else` branch
— which never calls `forward_tokens_verify` and never touches
`stats.draft_accepted`.** The missing `[verify-timing]` lines are a second,
independent gap: that print is gated on `DISMANTLE_QWEN_VERIFY_TIMING=1`, which
no script in the repo sets.

This is a **harness/wiring gap, not a kernel or token-space defect.** Type
classification per the Kill Protocol: **N/A — nothing died.** The fast verify
path is correct and bit-identical (gated by `354d718`); the draft simply was
not enabled in that invocation.

### Code chain that pins it

`crates/dismantle-core/src/lib.rs:32`
```rust
pub fn env_on(name: &str) -> bool {
    std::env::var(name).map(|v| v == "1").unwrap_or(false)   // unset → false
}
```

`qwen_dense.rs:1637-1640`
```rust
let use_user_draft = use_tcb
    && !use_eagle5
    && req.sampling.temperature == 0.0
    && crate::env_on("DISMANTLE_QWEN_USER_DRAFT");   // <-- false when unset
```

Branch dispatch in `generate()`:
- `qwen_dense.rs:1649` `if use_eagle5 { … }`
- `qwen_dense.rs:2229` `} else if use_user_draft { … 'ud_loop … }`  ← the draft loop
- `qwen_dense.rs:2394` `} else { … plain decode … }`  ← where the failing run went

In the plain branch (`2463-2552`) the per-step call is
`forward_token_greedy_tcb(last_id, pos)` (line `2470-2471`). `stats.draft_accepted`
is only ever incremented at `qwen_dense.rs:2328` (user-draft) and `1800`/`2102`
(eagle5). None of those run in the plain branch ⇒ `draft_accepted` stays its
init value `0`. And `forward_tokens_verify` (the only emitter of `[verify-timing]`,
`qwen_dense.rs:5271-5277`) is never called ⇒ zero timing lines. Both symptoms
fall directly out of "the `else` branch ran".

### Why `USER_DRAFT` was absent

`DISMANTLE_QWEN_USER_DRAFT` is **env-only — there is no CLI flag for it.**
`grep -rn USER_DRAFT crates/dismantle-cli/` returns nothing; it is read only via
`env_on` in `qwen_dense.rs`. `tools/bench/clean_room_batch.sh:63-67` exports the
fast-decode env but **not** `USER_DRAFT` (nor `VERIFY_TIMING`):
```sh
export DISMANTLE_QWEN_TCB=1 \
       DISMANTLE_QWEN_VOCAB_PRUNE=32000 \
       DISMANTLE_QWEN_Q4K_LMHEAD=1 \
       DISMANTLE_QWEN_FFN_DOWN_Q4K=1 \
       DISMANTLE_QWEN_Q4K_PREDEC=1
```
A bare `dismantle generate` under that env (or under the documented fast-decode
recipe) therefore takes the plain decode branch. `grep -rn 'USER_DRAFT|VERIFY_TIMING'
tools/ crates/dismantle-bench/` ⇒ empty: **no script anywhere enables either flag.**

---

## 2. Hypotheses considered and REFUTED by code logic

These were the candidate "Q4K-env-specific" causes. All are refuted, so the
draft path needs no token-space fix:

- **"VOCAB_PRUNE remaps the token stream so propose() feeds pruned ids while
  warm_start seeded real ids → permanent miss."** REFUTED. All token producers
  return **real** vocab ids:
  - bonus: `forward_token_greedy_tcb` remaps the GPU pruned-argmax index back to
    the real id before returning — `qwen_dense.rs:4578-4581`
    (`map.get(pruned_idx).unwrap_or(&pruned_idx)`).
  - verify preds (fast path): same remap — `qwen_dense.rs:5285`
    (`remap.map(|r| r[pi as usize])`).
  - verify preds (fallback): full-vocab argmax, already real ids — `5304-5307`.
  - warm-start: `draft_index.warm_start(&prompt_ids)` (`2251`), and `prompt_ids`
    are real ids from `self.tokenizer.encode` (`1348`).
  Index keys and propose context are all the same (real) space. No mismatch.

- **"The batched verify Stage-1 embeds pruned indices."** REFUTED. The batch
  embed uses the full-vocab `embed_buf` with the real `tok` —
  `qwen_dense.rs:4837-4842`.

- **"Prefix cache (default-on when the full env omits `PREFIX_CACHE=0`) skips
  prefill so the index is under-seeded."** REFUTED. The prefix cache only skips
  KV **recompute** (`prefill_skipped`, `qwen_dense.rs:1451`); `warm_start` always
  consumes the entire `prompt_ids` vector regardless (`2251`). Seeding is intact.

- **"`FFN_DOWN_Q4K` / `Q4K_PREDEC` break the draft loop."** REFUTED as a *break*.
  They perturb the verifier's logits, which lowers **acceptance** (`7 → 3`,
  `(measured, contaminated)`, findings-doc row 4 + read 2) — a real **sub-additive
  composition** — but the loop still runs and still emits `draft_accepted=3 > 0`.
  This is a quality-of-draft effect, not the cause of `draft_accepted=0`.

- **Earlier "eagle5 auto-load" theory.** REFUTED (and already known false):
  `eagle5_head` is `Some` only when `config.speculate_mode == Eagle5`
  (`qwen_dense.rs:1216`), and `use_eagle5 = use_tcb && eagle5_head.is_some()`
  (`1606`). With no `--speculate eagle5`, `use_eagle5 = false`, so the
  `!use_eagle5` term in `use_user_draft` is satisfied — eagle5 is not what
  suppressed the draft.

**Single root cause stands: the flag was off.** No NO-GO is recorded; a
CPU/weight-only proxy cannot and should not kill the draft path here.

---

## 3. Is the loop bonus-first? Is propose-first the right fix? — CONFIRMED

**Confirmed bonus-first = 2 target-forwards/cycle.** In `'ud_loop`
(`qwen_dense.rs:2258-2393`):
- **Stage 1** (`2269`): `bonus = self.forward_token_greedy_tcb(last_id, pos)` —
  forward #1, emits the true next token unconditionally.
- **Stage 2** (`2298`): `draft = draft_index.propose(&[ctx_prev, bonus], k_avail)`
  — CPU, free.
- **Stage 3** (`2318`): `forward_tokens_verify(&vtoks, &vpos)` — forward #2.

So every cycle pays **2 GPU forwards** and emits `1 + first_reject` tokens. At
zero acceptance that is 2 forwards for 1 token = **2× the plain-decode forward
cost per token** → the genuine low-acceptance slowdown (the regime behind any
paired regression on non-repetitive content). On a non-repetitive prompt
`propose()` returns empty (`draft_len == 0`, `2300-2305`) so it `continue`s
after the bonus — still **1 forward/token wasted on the propose attempt is
avoided**, but the *2-forward* cost lands the moment any draft is proposed and
fully rejected.

**Confirmed propose-first is the correct fix, and the eagle5 `use_propose_first`
loop is the right template.** In `'pf_loop` (`qwen_dense.rs:1766-1846`) there is
**no separate bonus forward inside the steady loop** — the single
`forward_tokens_verify(&vtoks, &vpos)` per cycle (`1787`) does double duty:
`vtoks = [carried_true, d1..d(k-1)]`, and `preds[0]` IS the correction / true
next token (`1797-1810`), so one forward emits `na+1` tokens (`1813` `for j in 0..=na`).
That is **1 forward/cycle**. The carried-true invariant (`1781`, `1842`
`carried_true = preds[na]`) is what removes the second forward.

**The user-ngram analog is strictly simpler** (the findings doc, §"one real next
lever", says the same): drop the residual machinery the head needs —
`read_res`/`anchor_res` (`1738-1745`, `1751`, `1841`) and the chained-hidden
`propose_rollout_chained` (`1777`) — and keep the pure 2-gram context
`(anchor_tok, carried_true)`. The n-gram's `propose(&[anchor_tok, carried_true], k)`
(`user_ngram.rs:155`) is a CPU lookup with no hidden state, so **no residual
read, no bootstrap-with-capture is required.** A *one-time* bootstrap forward to
get the first `carried_true` is still needed (eagle5 does it at `1750`); that is
1 forward total, amortized, not per-cycle.

### Net forward-count (code-logic, not a tps claim)

| loop | forwards/cycle | tokens emitted/cycle |
|---|---|---|
| current `'ud_loop` (bonus-first) | **2** (Stage-1 bonus + Stage-3 verify) | `1 + first_reject` |
| proposed propose-first | **1** (verify only) + 1 one-time bootstrap | `na + 1` (= accepted + correction) |

At acceptance `a` per cycle: current cost ≈ `2 / (1+a)` forwards/token;
propose-first ≈ `1 / (1+a)` forwards/token — **~2× fewer forwards per emitted
token in the limit**, with the gap largest exactly at low `a` (the regime that
can go net-negative today). This is an arithmetic statement about forward counts
`(code-logic)`; the realized tps is a **GPU paired-bench question** (§5).

---

## 4. The exact code change (file:line, before/after sketch)

New file, new flag — do **not** rewrite the existing `'ud_loop`. Add a sibling
guarded by `DISMANTLE_QWEN_USER_DRAFT_PROPOSE_FIRST` so the bonus-first loop
stays as the bit-identical reference for the gate.

**Gate (insert near `qwen_dense.rs:1641`, alongside `user_draft_k`):**
```rust
// before: (only user_draft_k exists)
let user_draft_k: usize = std::env::var("DISMANTLE_QWEN_USER_DRAFT_K") … .min(8);

// after: add the propose-first opt-in
let user_draft_k: usize = std::env::var("DISMANTLE_QWEN_USER_DRAFT_K") … .min(8);
let user_draft_propose_first =
    use_user_draft && crate::env_on("DISMANTLE_QWEN_USER_DRAFT_PROPOSE_FIRST");
```

**Branch split (at `qwen_dense.rs:2229`):**
```rust
// before:
} else if use_user_draft {
    // 'ud_loop  (bonus-first)  — qwen_dense.rs:2230-2393

// after:
} else if user_draft_propose_first {
    // ── NEW 'udpf_loop: propose-first, 1 verify forward/cycle.
    // Mirror of 'pf_loop (1734-1846) MINUS the head: no read_res,
    // no anchor_res, no propose_rollout_chained; draft source is
    // UserNgramDraft::propose(&[anchor_tok, carried_true], k).
    let mut idx = crate::speculate::user_ngram::UserNgramDraft::new();
    idx.warm_start(&prompt_ids);
    let k = user_draft_k.max(1);
    // bootstrap: one forward → first carried_true (no residual needed)
    let mut anchor_tok = last_id;
    let mut anchor_pos = prompt_len;
    let mut carried_true = self.forward_token_greedy_tcb(anchor_tok, anchor_pos)?;
    // … emit carried_true, idx.note_token, produced += 1, eos check …
    'udpf_loop: while produced < req.max_new_tokens
        && matches!(reason, StopReason::MaxTokens)
    {
        // propose chain from the 2-gram (anchor_tok, carried_true)
        let drafts = idx.propose(&[anchor_tok, carried_true], k); // d[0..]
        let dlen = drafts.len();
        if dlen == 0 {
            // no draft: do ONE plain forward to advance, like 'pf_loop's
            // degenerate case — anchor_tok ← carried_true; carried_true ←
            // forward(carried_true, anchor_pos+1); anchor_pos += 1; continue.
        }
        // verify [carried_true, drafts[0..dlen-1]] at anchor_pos+1 ..
        let mut vtoks = Vec::with_capacity(dlen + 1);
        vtoks.push(carried_true);
        vtoks.extend_from_slice(&drafts[..dlen.saturating_sub(1)]);
        let vpos: Vec<usize> = (0..vtoks.len()).map(|j| anchor_pos + 1 + j).collect();
        let (preds, _resids) = self.forward_tokens_verify(&vtoks, &vpos)?;
        // accept drafts[j] while drafts[j] == preds[j]; preds[0] is correction
        // … same accept/emit/usage_capture/eos bookkeeping as 'pf_loop:1790-1845,
        //    but drop residuals[na] (no anchor_res); set
        //    self.kv.seq_len = anchor_pos + 1 (+ na). anchor advances to the
        //    last accepted pos; carried_true = preds[na]; anchor_tok = the
        //    token consumed at that position.
    }
    // finalize (mirror 1847-1860: stats.decode_ms/completion_tokens, return)
} else if use_user_draft {
    // 'ud_loop unchanged (bonus-first) — kept as the parity reference
```

Implementation notes pinned to the eagle5 template so the KV/emit invariants are
copied, not re-derived:
- Accept/emit and KV-rewind bookkeeping: copy `qwen_dense.rs:1790-1845`. Replace
  the head-specific lines (`drafts[1+j]` indexing, `residuals[na].clone()` at
  `1841`, `head.note_token` at `1819`/`1759`) with the n-gram's
  `drafts[..]` indexing and `idx.note_token(id)`.
- `carried_true` invariant (never re-emitted; the previous cycle already emitted
  it) — `qwen_dense.rs:1781` + the `for j in 0..=na` at `1813`.
- Off-by-one: the user-ngram `propose` ctx is the 2-gram `(anchor_tok,
  carried_true)`; eagle5 uses the same 2-gram for `record_draft` (`1806`).
- `min(8)` verify-batch cap still applies (`forward_tokens_verify` guard `b ∈
  1..=8`, `qwen_dense.rs:5240`); cap `k` accordingly (eagle5 relies on
  `eagle5_k`; here use `user_draft_k.min(8)`).

This is the ~50-line restructure the findings doc named. It is a **hot-path
change behind a flag**, gated below, and must be **re-verified for token
identity by a human** before being trusted (CLAUDE.md worktree-parity rule).

---

## 5. Parity-coverage gap → exact test additions

**Current coverage (`crates/dismantle-core/tests/user_draft_parity_e2e.rs`):**
- `user_draft_is_bit_identical` — `TCB=1`, `PREFIX_CACHE=0`, default cfg →
  **CPU fp16 full-vocab fallback** verify. Asserts 16-token equality.
- `user_draft_bit_identical_fast_pruned_q4k` (added `354d718`) — `+ VOCAB_PRUNE=32000
  + Q4K_LMHEAD=1` → **GPU pruned-Q4K fast verify**; asserts 16-token equality
  **and `draft_accepted > 0`**.

> Correction to the task brief: the existing test file does **NOT** omit
> `Q4K_LMHEAD`. As of `354d718` the second test sets it and exercises the GPU
> fast path. The remaining gaps are (a) the **full** shipped env
> (`+ FFN_DOWN_Q4K + Q4K_PREDEC`), and (b) the **propose-first** loop. Both are
> uncovered today.

**Additions (all on the real Qwen-3B, skip-if-missing, serialized by the existing
`SERIAL_GATE`, mirroring the two current tests):**

1. `user_draft_bit_identical_full_fast_env` — set `VOCAB_PRUNE=32000`,
   `Q4K_LMHEAD=1`, `FFN_DOWN_Q4K=1`, `Q4K_PREDEC=1`, `USER_DRAFT=1` vs the same
   env with `USER_DRAFT=0`. Assert the 16-token vectors are equal. **Do NOT
   assert `draft_accepted > 0` here** — under FFN_DOWN_Q4K acceptance can legitimately
   be low (`3`, `(measured, contaminated)`); report it, don't gate on it. This
   closes the "the production fast-decode recipe is bit-identical with the draft
   on" gap that the failing CLI run never reached.

2. `user_draft_propose_first_bit_identical` — set `USER_DRAFT=1` +
   `USER_DRAFT_PROPOSE_FIRST=1`; reference is `USER_DRAFT=1` + propose-first OFF
   (the bonus-first `'ud_loop`). Assert the 16-token vectors are **identical to
   each other** (both must equal plain greedy by construction; testing them
   against each other pins that the restructure changed only forward-count, not
   tokens). Run it under **both** the default cfg and the pruned-Q4K cfg (two
   `#[test]`s, or a parametrized helper) so the new loop is covered on both the
   CPU-fallback and GPU-fast verify.

3. `user_draft_propose_first_lossless_long` — same as (2) but `MAX_NEW_TOKENS =
   64` on the repetitive prompt, to catch a KV-rewind off-by-one in the new loop
   that a 16-token window can miss (the rewind logic at the eagle5 `1839-1845`
   analog is the highest-risk part of the port).

Pattern to copy verbatim: `make_engine` + `gen_on` + env set/restore from
`user_draft_parity_e2e.rs:50-87` and the env-restore discipline at lines
`182-184` (remove the vars after, whatever order the tests run).

---

## 6. GPU-lane paired-bench protocol (to confirm the fix)

This is the **single decisive gate** — the only thing that settles whether
propose-first is a real tps win. It MUST run with Claude **quit** (the in-session
table in the findings doc is contaminated; absolute tps swings 1.3–30×,
[[bench_contamination]]).

1. **Correctness first (cheap, can run with Claude open):** the three new parity
   tests above green, 16- and 64-token vectors bit-identical. A perf bench on a
   non-bit-identical loop is meaningless.
2. **Quit Claude. Kill strays:** `pkill -f 'dismantle generate'` (the findings
   doc notes lingering paired-run processes hammered the GPU).
3. **Paired A/B, matched prompt + token budget**, using `paired_lever.sh` /
   `clean_room_batch.sh` env **plus `DISMANTLE_QWEN_USER_DRAFT=1`** (the flag the
   bench scripts currently omit — add it, see §1). Three arms:
   - **A** baseline: `USER_DRAFT=0` (plain fast decode).
   - **B** bonus-first: `USER_DRAFT=1`, propose-first OFF.
   - **C** propose-first: `USER_DRAFT=1`, `USER_DRAFT_PROPOSE_FIRST=1`.
4. **Two prompt classes**, ≥128 tokens each (the findings doc's owed
   measurement): (a) repetitive code (high n-gram acceptance), (b) natural prose
   / non-repetitive code (low acceptance — the regime where bonus-first's 2nd
   forward should hurt and propose-first should recover).
5. **Trials:** `TRIALS=20` per arm (matches the M5 stack-matrix discipline),
   report median + full spread, not the mean.
6. **Decision rule:**
   - `C > B` on the **low-acceptance** prompt by more than the inter-trial
     spread ⇒ propose-first removes the 2-forward penalty as predicted ⇒ keep it
     (consider default-on behind the flag).
   - `C ≈ B` on the **high-acceptance** prompt (expected: at high acceptance the
     bonus is mostly amortized either way) ⇒ no regression, fine.
   - `C < A` on **either** prompt ⇒ the draft is net-negative even propose-first
     ⇒ keep default-off; the lever is repetition-only.
7. **Instrumentation:** set `DISMANTLE_QWEN_VERIFY_TIMING=1` for arms B/C to get
   the `[verify-timing] B=.. fwd=.. gemm+commit=..` lines (`qwen_dense.rs:5271`),
   and read `draft_accepted` from the `Done` stats. Confirm B shows ~2× the
   forward count of C per emitted token (the §3 arithmetic) — that is the
   mechanism check independent of contaminated absolute tps.

**Bonus harness fix (out of scope to implement here, flag it):** add
`DISMANTLE_QWEN_USER_DRAFT=1` (and optionally `VERIFY_TIMING=1`) to
`tools/bench/clean_room_batch.sh` / the paired-bench scripts, or expose a
`--user-draft` CLI flag in `crates/dismantle-cli/`. Without one of those, any
future "does the draft help?" CLI run repeats the exact `draft_accepted=0`
non-result diagnosed here.

---

## 7. Verdict

- **Root cause (pinned):** `DISMANTLE_QWEN_USER_DRAFT` unset ⇒ `use_user_draft =
  false` ⇒ plain decode branch ⇒ `draft_accepted=0`, no `forward_tokens_verify`,
  no `[verify-timing]`. Compounded by `VERIFY_TIMING` also being unset. A
  harness/wiring gap; **the draft path itself is correct** (gated bit-identical,
  `354d718`). No lever died — **N/A, NEEDS-MEASUREMENT** for the propose-first
  tps question.
- **Loop is bonus-first (2 forwards/cycle):** confirmed (`qwen_dense.rs:2269` +
  `2318`).
- **Propose-first is the correct fix:** confirmed; the eagle5 `'pf_loop`
  (`1734-1846`) is the template, minus the residual/bootstrap-capture machinery
  the n-gram doesn't need. ~50 lines, behind `DISMANTLE_QWEN_USER_DRAFT_PROPOSE_FIRST`.
- **Decisive gate:** a clean-room (Claude-quit) 3-arm paired bench (A plain / B
  bonus-first / C propose-first) on a low-acceptance and a high-acceptance prompt
  at ≥128 tok, with `USER_DRAFT=1` actually exported. Until then propose-first is
  a forward-count win on paper, not a measured tps win.
