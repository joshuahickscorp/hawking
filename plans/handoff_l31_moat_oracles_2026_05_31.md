# Handoff — stateful-moat L3.1 first move: 3 NumPy oracles (kill-or-keep), then `usage_capture`

> Paste this whole file as the opening prompt for a FRESH Claude Code session
> (local or cloud). It is self-contained. This work is **offline / pure NumPy /
> CPU — zero GPU, no model decode, no kernel.** Another session may be running on
> the GPU (a prefill-MMA kernel build); this lane touches **disjoint files** and
> the GPU is never contended. Branch: `codex/maximal-spec-colab`. Commits are
> authored `Joshua Hicks <joshuahicksboba@gmail.com>` inline with **NO AI
> attribution** (no `Co-Authored-By`, no "Generated with" — the message must look
> human-authored). Do NOT push; commits stay local.

## What dismantle is
A from-scratch Rust + Metal single-stream inference engine for Qwen2.5-3B-Q4_K_M
on an M3 Pro (18 GB). Strategy doc: `plans/throughput_bible_2026_05_30.md`. Read
**§3.0** (status correction) and **§8** (the System-Level Shift) first.

## Why this work now
Decode is at its kernel ceiling — bible §3.0: the decode-GEMV micro-opt track is
CLOSED (~31 clean tps, at the Apple-GPU memory-model optimum). The durable
frontier is now §8's **stateful moat** ("the model molds to its owner"). This is
its first concrete move. **The full design spec is
`plans/stateful_moat_continuation_design_2026_05_31.md` — READ IT; this handoff is
the execution wrapper around its §3 "first move."**

## The task — run 3 oracles FIRST (the gate), build bodies ONLY if one clears
The project discipline (bible §8.3, CLAUDE.md Kill Protocol): **no body is written
until its named offline oracle clears.** The 3 oracles below run on **existing
data** (the git-history session corpus + the dequantized GGUF lm_head) — they do
**NOT** need the `usage_capture` observer first. Build the observer + a lever body
only in Phase 1, after an oracle greenlights. Do them in this confidence order:

### Oracle 1 — draft tuning (HIGHEST confidence; build first if it clears)
- **Extend `tools/bench/oracle_spec_accept.py`** (it currently measures generic
  PLD/n-gram τ = **1.43** on code; verdict in `reports/oracle/spec_accept.json`).
  Read it to find the corpus it already consumes; reuse that.
- **Measure:** a *user-warm-start* variant — build the n-gram / suffix-automaton
  draft index from a **prior slice** of a session's token stream, then measure
  accepted-length τ on a **held-out later slice** of the *same* session (per-session
  specialization is the per-user proxy). Compare to the τ=1.43 generic baseline.
- **Greenlight:** warm τ lifts **materially above 1.43 toward the 2.5 gate** — even
  τ≥1.8 is a real win (the draft is free CPU + lossless; unlike the dead EAGLE head
  there is NO net-negative risk).
- **Honest prior / risk:** the user's repeats may already be captured by in-prompt
  PLD → no lift → NO-GO. Cheap to find out.

### Oracle 2 — semantic cache (MEDIUM confidence)
- **Extend `tools/bench/oracle_prefix_cache.py`** (it already emits
  `semantic_cache.near_dup_rate_cosine` / `verified_near_dup_rate_cosine` and a
  `hashed_bigram_vec`). Corpus: `tools/bench/make_git_session_corpus.py` (already
  used; see `reports/oracle_prefix_cache_githistory.md`, ~45% mean / 13–72% session
  *consecutive* exact reuse).
- **Measure:** the **incremental** reuse the semantic tier buys *over* the exact
  tier — per request, (b) longest exact-common-prefix with *any* prior context whose
  embedding cosine ≥ τ_sem, minus (a) exact-consecutive reuse. Sweep τ_sem ∈
  {0.95, 0.90, 0.80, 0.70} and `MIN_REUSE_TOKENS`.
- **Greenlight:** semantic-augmented session-reuse beats exact-only by **≥ ~10
  percentage points** at a τ_sem with **≥95% verify-confirm rate** (precise
  retrieval, not noisy).
- **Honest prior / risk:** non-consecutive near-dups may be rare (the exact tier
  already wins the focused-edit loop) → modest uplift → record, don't build.

### Oracle 3 — vocab screen (LOWEST confidence — L-M; do NOT build before it clears)
- **New `tools/bench/oracle_vocab_coverage.py`**, extending
  `oracle_svd_lmhead.py`'s GGUF lm_head dequant loader (reuse it to get per-row
  weights → `‖w_v‖`).
- **Measure:** (a) effective vocab coverage — what fraction of 151,936 is ever the
  argmax, and the hot-set size H covering ≥99.x% of argmax steps; (b) **norm-bound
  certificate fall-back rate** — simulate the screen: an out-of-H token is provably
  not the argmax iff `‖w_v‖·‖h‖ < ℓ_c`; count how often *all* out-of-H tokens
  satisfy the bound (certified fast path) vs require a full pass. Use real
  hidden-state norms if available, else a modeled `‖h‖` distribution (**label it
  estimate**).
- **Greenlight:** a small H (≤ a few K tokens) gives **≥80% certified fast path**
  (the byte cut is real, fall-back rare).
- **Honest prior / risk:** the lm_head is **FULL-RANK** (rank99 = 1987/2048,
  cond ≈ 45, similar row norms) — exactly the regime where a norm bound is **weak**
  and rarely certifies — and the head is only ~4–10% of bytes/token. Expect it to
  **die cheaply** at this oracle. Include it only because its mechanism (frequency +
  certificate) is genuinely different from the dead SVD screen.

### Phase 1 — ONLY after an oracle clears
Build `crates/dismantle-core/src/stateful/usage_capture.rs` — a parity-neutral,
default-off side-observer **copying `attn_capture.rs` exactly** (`Mutex<Option<…>>`,
`enabled()`, record hooks at the emit sites in `qwen_dense.rs`, JSON flush). Then
the cleared lever's body per the design doc (§1.5 semantic / §2.3 vocab+draft).
**The observer must add ZERO hot-path cost when off and change NO token output —
assert parity.** Order: draft-tuning (B-b) → semantic (A) → vocab-screen (B-a).

## The kills that bound this — do NOT re-litigate (Kill Protocol: never re-test a Type-1)
From `reports/dead_levers.md` + design §0:
- **lm_head SVD screen: FULL-RANK NO-GO** → the vocab prune is **usage-frequency +
  norm-certificate, NOT a rank screen.** (Why Oracle 3 is low-confidence.)
- **EAGLE-3 trained head: NO-GO** (τ=0.877 held-out, net-negative on device) → draft
  tuning targets the **n-gram / SAM** draft, **NOT a learned head.** Do not propose
  training a head.
- **L1.1 KV working-set: NO-GO** (attention mass diffuse) → parked; not in scope.
- Generic n-gram spec baseline = **τ 1.43** (the number Oracle 1 must beat).

## Method + constraints
- **Pure NumPy / CPU.** No GPU, no kernel, no model decode. The GGUF lm_head dequant
  for `‖w_v‖` is CPU NumPy (as `oracle_svd_lmhead.py` already does). Runs
  parallel-safe with any GPU session.
- **Kill Protocol (CLAUDE.md / bible §8.3.1):** any NO-GO must record Type-1 vs
  Type-2 + the reframe considered + why it dies / its next oracle, in
  `reports/dead_levers.md`.
- **Report spread, not just the mean; label estimate vs measured.** The corpus is a
  *proxy* for a real user (label estimate); the production number comes from
  `usage_capture` logs later (measured). Carry the full range.
- **Do NOT touch:** `git stash@{0}` (a half-built prefill-MMA kernel — another
  lane's WIP), the kernel files (`shaders/quant.metal`, `kernels/mod.rs`), or the
  untracked junk (`on_smoke.err`, `server_smoke.err`, `traces/`).
- **If you write the observer (Phase 1):** `cargo build --release --workspace` +
  `cargo test -p dismantle-core --lib` must pass; the observer is **parity-neutral,
  default-off.**
- Commit oracle scripts + results (`reports/oracle/*.json`, a short findings `.md`)
  **separately** from any source body. One purpose per commit. Inline git identity,
  no AI attribution. Don't push.

## Output
For each of the 3 oracles: **GO / NO-GO** with the measured number (+ spread, +
estimate/measured label) and the greenlight comparison. Then: the **3 levers ranked
by measured payoff**, and a one-line recommendation on which body (if any) to build
first. Lead with Oracle 1 (draft tuning). Keep the writeup under ~500 words; commit
the raw oracle JSON for the audit trail.

## Pointers (all verified present 2026-05-31)
- Design spec: `plans/stateful_moat_continuation_design_2026_05_31.md`
- Oracles to extend: `tools/bench/oracle_spec_accept.py`,
  `tools/bench/oracle_prefix_cache.py`, `tools/bench/oracle_svd_lmhead.py`
- Corpus generator: `tools/bench/make_git_session_corpus.py`;
  prior result `reports/oracle_prefix_cache_githistory.md`
- Bounding kills + results: `reports/dead_levers.md`,
  `reports/oracle/spec_accept.json`, `reports/oracle/svd_lmhead.json`
- Observer template: `crates/dismantle-core/src/stateful/attn_capture.rs`
- Landed infra the bodies build on: `crates/dismantle-core/src/stateful/prefix_cache.rs`
  (`SemanticIndex` stub ~696), `crates/dismantle-core/src/vocab_prune.rs`,
  `crates/dismantle-core/src/speculate/shared.rs`
