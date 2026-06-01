# Draft-tuning verify path — measured findings (2026-05-31)

> Opened against MOVE 2 of the evening handoff: "the pruned-Q4K batched-verify
> fast-path … Draft tuning is −78% when on because `forward_tokens_verify` falls
> back to a CPU full-vocab pass at ~18% acceptance. Make the K-wide verify GEMM
> use the pruned-Q4K LM head." **That premise is falsified by measurement.** This
> doc records what is actually true, what was built, and the one real next lever.

## TL;DR

- The **GPU pruned-Q4K batched-verify fast path already exists** (`010827b`,
  2026-05-29 — it *predates* the draft-tuning body `680cb35`, 2026-05-31). It is
  wired into `forward_tokens_verify` (`qwen_dense.rs:5235`) and used by BOTH the
  eagle5 and the user-ngram draft loops. It is NOT missing.
- It **engages** whenever the shipped fast-decode env builds a Q4_K pruned head
  (`DISMANTLE_QWEN_VOCAB_PRUNE` + `DISMANTLE_QWEN_Q4K_LMHEAD` → `vocab_pruned_is_q4k`
  true; the verify batch `b = draft_len ≤ 8` always satisfies the `1..=8` guard).
  `clean_room_batch.sh` sets exactly this env.
- It is **bit-identical** to plain greedy — now **gated** by a new parity test
  (`user_draft_bit_identical_fast_pruned_q4k`). The pre-existing gate only covered
  the **CPU fp16 full-vocab fallback** path (it does not set `_Q4K_LMHEAD`), the
  very path `010827b` says **diverged** — so the production fast path was
  un-gated until now. Both gates green, `draft_accepted=7`, identical token vector.
- The cited **−78% is a wrong-env / low-acceptance artifact, not a property of
  the fast path.** With the fast path engaged on the n-gram's target workload
  (repetitive code), draft tuning is strongly **positive**, not negative.

## What was measured (in-session; contamination-noted)

In-session benches are contaminated (absolute tps swings ~1.3–30 between runs as
my session loads the GPU; [[bench_contamination]]). Stray backgrounded `dismantle
generate` processes from earlier paired runs made it worse (they lingered and
hammered the GPU; killed). So **absolute tps below is indicative only** — the
clean paired delta is owed to a clean-room run (Claude quit). What IS reliable:
token-identity (deterministic) and the relative direction at matched contention.

Repetitive prompt (`fn add(a: i32, b: i32) -> i32 { a + b }` ×3), 16 tokens,
USER_DRAFT_K=6, a low-contention window:

| env | dec_tps | draft_accepted |
|---|---|---|
| TCB only (minimal) | 12.2 | 7 |
| + VOCAB_PRUNE | 12.5 | 7 |
| **+ VOCAB_PRUNE + Q4K_LMHEAD** (fast verify) | **30.2** | 7 |
| full env (+ FFN_DOWN_Q4K + Q4K_PREDEC) | 25.9 | **3** |

Reads:
1. **The fast verify path is the win**: adding `Q4K_LMHEAD` (which makes the
   verify a GPU `gemm_q4_k_m_batched_v3w` over the pruned head instead of a CPU
   full-vocab pass) jumps 12.2 → 30.2 tps on this prompt. That is the lever the
   handoff wanted "turned into real tps" — and it was already there.
2. **`FFN_DOWN_Q4K` lowers draft acceptance** 7 → 3. Quantizing the FFN-down
   projection perturbs the logits enough that the n-gram's drafts (an automaton
   grown from the un-perturbed stream) get rejected more — a real **sub-additive
   composition** between the ffn-down byte-cut and the draft. It still cuts raw
   decode bytes, so the NET draft-tuning tps under the full shipped env needs the
   clean paired bench to call.
3. On **non-repetitive** prompts the n-gram rarely proposes (`draft_accepted=0`),
   so draft-on ≈ draft-off minus a little bookkeeping. The draft is a *code /
   repetition* lever, not a general one — consistent with the τ≈3.40 warm-start
   oracle being measured on the user's own repeated tokens.

## What was built this session

- `crates/dismantle-core/tests/user_draft_parity_e2e.rs`:
  `user_draft_bit_identical_fast_pruned_q4k` — gates the FAST pruned-Q4K verify
  path bit-identical (asserts `draft_accepted > 0` so it cannot pass vacuously).
  Closes the coverage gap (the bench uses the fast path; nothing gated it).
- This findings doc (premise correction + measured reality + next lever).

Nothing in `forward_tokens_verify` needed changing — the fast verify GEMM already
uses the pruned-Q4K LM head. The handoff's "make the K-wide verify GEMM use the
pruned-Q4K LM head" was already done in `010827b`; the action was to *verify* it
(done, gated) and correct the −78% story.

## The one real next lever (named, not built)

**Propose-first restructure of the user-ngram loop** (`qwen_dense.rs:2229` `'ud_loop`).
It is currently **bonus-first**: per cycle it runs a Stage-1 bonus forward AND a
Stage-3 verify forward = **2 forwards/cycle**, emitting `1 + accepted` tokens. At
low acceptance that is net-negative (the regime that can produce a paired
slowdown on non-repetitive content). The eagle5 path already solved this with
`use_propose_first` (`qwen_dense.rs:1734`): verify the full chain `[carried_true,
d1..dK]` in ONE batched forward whose `preds[0]` IS the correction — **1
forward/cycle**, no separate bonus. The user-ngram analog is *simpler* (no head,
no residual capture/bootstrap — drop the `read_res`/`anchor_res` machinery, keep
the `(anchor_tok, carried_true)` n-gram context). Estimated ~50 lines mirroring
`'pf_loop`. **Gate it the same way** (extend the two parity tests to the
propose-first env) and **re-verify token-identity** before trusting it.

Why not built now: it is a hot-path restructure whose payoff is exactly the
low-acceptance regime, and that payoff cannot be cleanly measured in-session
(contamination). Build it attended, behind its own flag, with the clean paired
bench as the gate.

## Owed measurement (clean-room, Claude quit)

Paired draft OFF vs ON under `clean_room_batch.sh`'s env on (a) a repetitive-code
prompt and (b) a natural-code prompt, ≥128 tokens. That is the only way to get
the honest net draft-tuning tps (including the `FFN_DOWN_Q4K` acceptance hit).
Until then: the fast verify path is **proven correct** (gated bit-identical) and
**indicatively positive** on its target workload; the net headline tps is
deferred, not claimed.
