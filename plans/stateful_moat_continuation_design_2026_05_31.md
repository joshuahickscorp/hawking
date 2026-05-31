> **STATUS: DESIGN-AHEAD — reviewable mechanism + interface + oracle, not a wired implementation.**
> Companion to `plans/throughput_bible_2026_05_30.md` §8 ("The System-Level Shift") and to
> `plans/stateful_core_design_2026_05_30.md` (the L1.1/L1.2 interface scaffold). Scope: the **next
> two stateful-core / moat levers** now that decode-tps is at its kernel ceiling (bible §3.0 — the
> decode GEMV micro-opt track is closed) and the durable differentiated axis is the primary frontier:
>   1. **L1.2 extension — SEMANTIC caching** atop the just-landed exact prefix cache.
>   2. **L3.1 — online vocab / draft specialization.**
> Both designs build on landed infra (`crate::stateful::prefix_cache`, `crate::vocab_prune`,
> `crate::speculate`) and follow the project discipline: a mechanism, an exactness/quality
> guarantee, a build plan against the real files/APIs, a cheap offline oracle that must clear
> before any body is written, and an Apple-Silicon feasibility verdict. **No source edits, no GPU,
> no commits in the haul that produced this doc.**

---

# `dismantle` — Stateful Core continuation: L1.2 semantic cache + L3.1 vocab/draft specialization

*This doc is the design half. It commits nothing to the forward pass. Each lever's bodies are written
only after its named oracle clears — the same gate that killed block-256 sparsity, L1.1 KV
working-set, L1.3 cross-layer delta, L1.4 low-rank, L1.5 codebook, and the EAGLE-3 trained head.*

---

## 0. What already exists (do not rebuild) — and what just changed

The two levers below extend **landed**, not hypothetical, infrastructure. The relevant facts as of
2026-05-31:

| existing artifact | file | role for this design |
|---|---|---|
| **Exact prefix cache (L1.2 core) — LANDED + default-on** | `crates/dismantle-core/src/stateful/prefix_cache.rs` | `InMemoryPrefixCache` (bit-identical KV reuse, rolling-prefix-hash key, ~3 GiB byte-capped LRU). The semantic layer (L1.2 extension) sits *in front of* this and **reuses its restore path verbatim**. |
| **Prefix-cache decode wiring** | `crates/dismantle-core/src/model/qwen_dense.rs` ~1361–1420 (lookup/restore) + ~1540–1557 (insert) | `ram_prefix_cache: Option<InMemoryPrefixCache>` field; default-on via `env_opt_out("DISMANTLE_QWEN_PREFIX_CACHE")`. `lookup_counting → restore_into` at the lookup seam, `insert_from_kv` at the store seam, disk tier behind it. **The semantic probe slots into this exact seam.** |
| **`SemanticIndex` trait stub** | `prefix_cache.rs` ~696–702 (`// SEMANTIC LAYER (later)`) | `index(key, &[f32])` / `probe(&[f32], threshold) -> Vec<SemanticCandidate>`. Reserved seam; this doc fills in the embedding, the index, and the verify flow. |
| **`PrunedVocab` (static vocab prune) — LANDED** | `crates/dismantle-core/src/vocab_prune.rs` | Load-once whitelist, `pruned_to_original` / `original_to_pruned`, `slice_lm_head_f16`. **Explicitly NOT exact**: works only "when the top-1 token is in the whitelist" (see its `argmax_round_trip_via_pruned_space` test) and "no hot-reload." L3.1 closes both gaps: a certifiable screen for exactness + dynamic adaptation. |
| **Speculation verify primitives — LANDED** | `crates/dismantle-core/src/speculate/shared.rs`; `qwen_dense.rs` `forward_tokens_verify` (~1764), `propose_rollout_chained`, `note_token` | `verify_window` / `verify_draft_ids_until_mismatch` (longest-agreeing-prefix, lossless). The propose→batched-verify→accept loop already runs (`DISMANTLE_QWEN_EAGLE5_PROPOSE_FIRST`). **L3.1 draft tuning plugs the user-specialized draft into this loop; the verify stays unchanged so output stays exact.** |
| **`attn_capture` instrument (the oracle pattern)** | `crates/dismantle-core/src/stateful/attn_capture.rs` | Flag-gated (`DISMANTLE_QWEN_ATTN_CAPTURE`), process-global `Mutex<Option<…>>` side-observer, JSON to `$…_OUT`, read by an offline `tools/bench/oracle_*.py`. **This is the exact template for the L3.1 vocab-coverage capture instrument.** |
| **Oracle family** | `tools/bench/oracle_prefix_cache.py`, `oracle_spec_accept.py`, `oracle_svd_lmhead.py` | CPU/NumPy, no kernel. The L1.2-semantic oracle extends `oracle_prefix_cache.py` (it already computes a near-dup rate); the L3.1 oracles extend `oracle_svd_lmhead.py` / `oracle_spec_accept.py`. |

**Three landed results that bind these designs (do not re-litigate):**

1. **lm_head SVD screen is FULL-RANK NO-GO** (`reports/oracle/svd_lmhead.json`: rank99 = 1987/2048
   = 97% of dim; tied `token_embd.weight`, Q6_K, vocab 151936). *Consequence:* L3.1 vocab pruning
   **must not** be a low-rank screen — that mechanism is dead. It is a **usage-frequency** prune,
   a different and simpler mechanism, and its exactness guarantee is structural (always-include-the-
   true-argmax / fall back on miss), **not** a rank bound. This is called out explicitly in the task.
2. **n-gram / PLD speculation on code is τ≈1.43** (`reports/oracle/spec_accept.json`: best warm
   τ=1.427 at n=2/K=16; verdict NO-GO vs the 2.5 gate; hit rate 15.7%; bimodal acc_hist — 85% of
   steps accept 0, hits accept long runs). *Consequence:* L3.1 draft tuning's job is to **raise τ
   from 1.43 toward the gate by specializing the draft to the user's repeats** — it is the only
   remaining live spec mechanism (see #3).
3. **EAGLE-3 trained head is NO-GO, doubly confirmed** (`reports/dead_levers.md`: τ=0.877 held-out,
   net-negative on device 0.40×/0.30×/0.21× at K=2/4/8; "the free n-gram draft τ=1.43 beats the
   trained head 0.877"). *Consequence:* L3.1 draft specialization targets the **n-gram/SAM draft**,
   not a learned head. Do not re-propose training a head.
   **Also dead: L1.1 KV working-set** (`dead_levers.md`: attention mass diffuse on 586-tok code,
   99% mass needs 78–92% of positions). So the two levers here — semantic cache + vocab/draft — are
   the correct *live* continuation of the stateful core; L1.1 is parked behind a longer-context
   re-capture.

---

## 1. Lever A — Semantic caching (L1.2 extension)

### 1.1 Lever mapping

- **§8 lever:** L1.2 — cross-prompt computation reuse, the *semantic* half. Bible §8.1 L1.2:
  "extend past exact-match into semantic caching: embed recent contexts and recognize 'I have
  computed something near-identical before → reuse,' **with an exact-match verification step before
  trusting a near-hit**."
- **Mechanism:** the exact prefix cache only fires when a new prompt is a byte-exact token-prefix
  extension of a cached one. A user who re-opens the same file with one line changed, or re-orders
  imports, or re-sends a context with a different leading comment, **breaks the exact prefix** at the
  first differing token and recomputes everything after it — even though 95%+ of the KV is reusable.
  Semantic caching recognizes the near-duplicate, finds the **longest exact common prefix** with that
  near-duplicate (which the exact-match key alone would not surface because the *full-prompt* hashes
  differ), and reuses that prefix's KV. The reuse itself is still the bit-identical exact-prefix
  restore — **semantics only widens the candidate set the exact restore is applied to.**
- **Benefit:** converts the prefix-cache hit-rate from "consecutive request is a strict extension"
  (the current default-on behavior) to "any prior session context that shares a long prefix." The
  git-history oracle measured ~45% mean / 13–72% session reuse for *consecutive* exact prefixes
  (`reports/oracle_prefix_cache_githistory.md`); the semantic layer's ceiling is the **near-dup
  rate**, which the same oracle already emits (`semantic_cache.near_dup_rate_cosine` /
  `verified_near_dup_rate_cosine`).
- **Exact?** **E (greedy-lossless) in the default mode**, by construction — see §1.4. A near-hit is
  never trusted on similarity alone; it only nominates a candidate whose **exact common prefix** is
  then restored. A separate, clearly-labeled `SemanticMode::QualityBounded` (reuse the near-hit's KV
  past the exact prefix) is **Q** and **off by default** — it is for an explicitly opt-in
  quality-traded fast path, never greedy-exact.
- **Energy verdict:** **GENUINE** (a skipped forward pass is power not drawn; same as exact prefix).

### 1.2 The embedding method (cheap, local, model-free — no second model load)

The hard constraint from the task: prefer a cheap local embedding, **NOT a second model load if
avoidable.** Three candidates, ranked; the design picks #1 and reserves #2.

1. **Hashed n-gram sketch (CHOSEN).** Exactly the `hashed_bigram_vec` already in
   `tools/bench/oracle_prefix_cache.py` (unigram + bigram counts hashed into a fixed-width sparse
   vector; cosine over them approximates n-gram overlap). It is **model-free, allocation-light,
   computed on the token-id stream the engine already has**, and — decisively — it is **the same
   signal the oracle measures**, so the production index and the offline oracle agree on what
   "near-duplicate" means. For coding contexts (which share long literal token runs: imports,
   function bodies, file headers), n-gram cosine is a strong proxy for prefix reusability. A fixed
   1024- or 2048-dim hashed bag (MinHash-style or count-sketch) makes the per-context embedding a
   few-KB dense vector; cosine is a dot product.
2. **Reuse an already-resident signal (RESERVED).** The model's own early-layer mean-pooled hidden
   state (e.g. the residual at the capture layer the eagle5 path already pins,
   `eagle5_capture_residual_buf`) is a "free" learned embedding — it exists during prefill at zero
   marginal cost. Reserved as a *quality* upgrade if the hashed-n-gram recall proves too low in the
   oracle; it does not need a second model, only a tap into a buffer that is already produced.
   **Not** the default: it couples the embedding to the GPU forward and complicates the
   pure-bookkeeping property.
3. **A tiny dedicated embedding model (REJECTED).** A second model load contradicts the task's
   constraint and the bible's energy thesis (it adds bytes to the bus and a load-time cost). Only
   revisit if both #1 and #2 fail the recall oracle — and even then it is a separate lever, not this
   one.

**Decision: ship #1 (hashed n-gram). It needs no GPU, no model, and is oracle-faithful.**

### 1.3 The index, the near-hit → verify → reuse flow, and eviction

The index is small enough to live in unified memory next to the prefix cache. Design it as the body
of the **already-stubbed `SemanticIndex` trait** (`prefix_cache.rs` ~696):

```
trait SemanticIndex {
    fn index(&mut self, key: PrefixKey, embedding: &[f32]);
    fn probe(&self, embedding: &[f32], threshold: f32) -> Vec<SemanticCandidate>;
}
// SemanticCandidate { candidate: PrefixKey, similarity: f32 }   // already defined
```

- **Storage:** a bounded `Vec<(PrefixKey, EmbeddingSketch)>` (or a flat `f32` matrix of N×D for a
  vectorized cosine). N is small — one entry per *retained prefix-cache entry* (the same entries
  `InMemoryPrefixCache` already holds, default cap ~hundreds), so the index is **co-bounded with the
  prefix cache** and adds only N×D×4 bytes (e.g. 256 entries × 2048 dims × 4 B ≈ 2 MB). It shares the
  prefix cache's lifecycle: an entry is indexed when `insert_from_kv` lands its KV, and **evicted in
  lockstep** when the prefix cache's LRU drops that `PrefixKey` (so the index can never reference a
  prefix whose KV is gone). No separate eviction policy — it is a derived view of the prefix cache's
  key set.
- **Flow at the lookup seam** (`qwen_dense.rs` ~1386, *after* the exact `lookup_counting` misses):

```
1. exact:    cache.lookup_counting(key, &prompt_ids)         ← LANDED; if hit, done (this is the common path)
2. semantic: if miss AND semantic_on:
     emb        = hashed_ngram_sketch(&prompt_ids)            ← cheap, CPU
     candidates = index.probe(&emb, SEM_THRESHOLD)            ← cosine ≥ τ_sem, ranked
     for cand in candidates (highest similarity first):
       n = exact_common_prefix_len(cand.tokens, prompt_ids)   ← the VERIFY: real token compare
       if n >= MIN_REUSE_TOKENS:
         cache.restore_into(&cand_key, &prompt_ids, n, &mut self.kv)   ← LANDED restore, bit-identical
         ram_prefill_skipped = n; break
3. else fall through to disk tier                              ← LANDED
```

  The **verify is an exact token-prefix comparison**, not a similarity acceptance. A near-hit only
  earns a re-use if it genuinely shares a real common prefix of ≥ `MIN_REUSE_TOKENS` (e.g. 32) — and
  what gets reused is *only that exact common prefix's KV*. The decode then prefills
  `prompt_ids[n..]` exactly as today. **The similarity score is a retrieval heuristic; the token
  compare is the correctness gate.**
- **Why this finds reuse the exact cache misses:** the exact `lookup_counting` walks the rolling
  hash of `prompt_ids` and matches only prefixes *of the current prompt* that were inserted under
  their own full-prompt key. If prompt B = "X" + edited("Y"), and the cache holds A = "X" + "Y", the
  exact lookup matches the "X" prefix of B **only if "X" was itself ever inserted as a standalone
  entry.** When A was inserted whole, the exact tier can still find the shared "X" run via the
  rolling-hash walk — *but only up to the first divergence and only against entries keyed on a prefix
  of B*. The semantic tier widens this: it nominates A as a candidate even though A's full hash ≠ B's,
  then the token-compare recovers the exact shared "X". The net new reuse is **near-duplicate
  contexts that are not consecutive strict extensions** — the case the consecutive-pair oracle
  cannot count but the near-dup-rate metric can.

### 1.4 The exactness / quality guarantee

**Default mode (`SemanticMode::ExactVerified`) is greedy-lossless — formally identical to the exact
prefix cache's guarantee:**

1. The KV for tokens `[0..n)` is a pure function of `(model, tokenizer, tokens[0..n))` (decode is
   causal). This is the exact-cache theorem already proven in `prefix_cache.rs` and asserted by
   `cold_vs_warm_prefill_byte_identical`.
2. The semantic layer **only ever calls `restore_into` with `n = exact_common_prefix_len(candidate,
   prompt_ids)`** — i.e. a length over which the candidate's tokens are *byte-identical* to the
   current prompt's. So the restored KV bytes equal what a cold prefill of `prompt_ids[0..n)` would
   produce.
3. Therefore semantic reuse changes **nothing** about the logits at any position ≥ n; greedy argmax
   is identical; output is bit-identical. The similarity threshold cannot introduce error because it
   never decides what bytes are reused — only which candidate to *check*. A spurious near-hit that
   shares no real prefix yields `n < MIN_REUSE_TOKENS` and is rejected; the worst case is a wasted
   cheap token-compare, never a wrong reuse.
4. The same three invariants the exact tier honors carry over verbatim: never reuse across a
   `model_hash`/`tokenizer_hash` change (the `PrefixKey` binds both); never reuse the full prompt
   (`restore_into` already bails one token short via the matched-len rule); inserted blocks are
   immutable.

**Quality-bounded mode (`SemanticMode::QualityBounded`, OFF by default, opt-in flag):** reuse the
near-hit's KV *past* the exact common prefix (accept the approximation that the divergent tail's KV
is "close enough"). This is **Q**, clearly labeled, and gated by its own quality oracle (a
logit-cosine / token-divergence measurement, the same metric class the W4A8 quality work needs). It
is documented here for completeness and to reserve the enum variant, but **it is not part of the
greenlight for the default lever** — the default ships exact-verified or not at all.

### 1.5 Build plan (files / APIs, building on the prefix-cache infra)

- **`crates/dismantle-core/src/stateful/prefix_cache.rs`** — implement the `SemanticIndex` trait
  body as a new `InMemorySemanticIndex` struct (bounded `Vec` co-evicted with the prefix cache).
  Add `hashed_ngram_sketch(tokens: &[u32]) -> Vec<f32>` (port `hashed_bigram_vec` from the oracle,
  fixed-width). Add `SemanticMode { ExactVerified, QualityBounded }` and `SEM_THRESHOLD` /
  `MIN_REUSE_TOKENS` consts. Add a `semantic_lookup(&self, emb, prompt_ids) -> Option<(PrefixKey,
  usize)>` helper that does probe → token-compare → returns `(candidate_key, n)`. **Reuse
  `restore_into` and `insert_from_kv` unchanged.** Unit tests mirror the existing ones: a
  near-duplicate (1 token changed mid-prefix) must yield the correct exact-common-prefix `n` and a
  byte-identical warm prefill (`assert_kv_eq` against a cold prefill of `prompt_ids[0..n)`); a
  spurious high-cosine/low-prefix pair must be rejected (`n < MIN_REUSE_TOKENS`).
- **`crates/dismantle-core/src/model/qwen_dense.rs`** — at the lookup seam (~1386), after the exact
  `lookup_counting` miss and before the disk tier, add the semantic probe behind
  `env_opt_*("DISMANTLE_QWEN_SEMANTIC_CACHE")` (suggest **opt-IN** initially, promote to opt-out
  after the oracle + a paired bench). Index the prompt's sketch alongside the existing
  `insert_from_kv` at the store seam (~1556). Hold `ram_semantic_index: Option<InMemorySemanticIndex>`
  next to `ram_prefix_cache`. **One reviewable diff, two seams, mirrors the exact-cache wiring.**
- **No kernel, no GPU, no new dependency.** All CPU bookkeeping over existing arenas.

### 1.6 Oracle / gate to greenlight it

- **Oracle:** extend `tools/bench/oracle_prefix_cache.py` (it already computes
  `semantic_cache.near_dup_rate_cosine`, `verified_near_dup_rate_cosine`, and
  `mean_best_prior_cosine`). The new measurement the *body* needs is the **incremental reuse the
  semantic tier buys over the exact tier**: for each request, compute (a) exact-consecutive reuse
  (already done) and (b) **best-prior exact-common-prefix reuse** = the longest exact common prefix
  with *any* prior context whose embedding cosine ≥ τ_sem (not just the immediately preceding one).
  The delta (b) − (a), aggregated, is the semantic lever's payoff. Sweep τ_sem ∈ {0.95, 0.90, 0.80,
  0.70} (the oracle's existing `SEM_THRESHOLDS`) and `MIN_REUSE_TOKENS`.
- **Greenlight threshold:** build the body when the oracle shows **`verified_near_dup` incremental
  reuse adds a material fraction over exact-consecutive reuse** — concretely, when the
  semantic-augmented session-reuse-fraction beats the exact-only fraction by ≥ ~10 percentage points
  at a τ_sem with **≥95% verify-confirm rate** (so the retrieval is precise, not noisy). If
  near-duplicates that are not consecutive extensions are rare on real transcripts (the exact tier
  already catches the focused-edit loop), the lever is **Type-1-ish dead for now** — record it,
  don't build it. The honest prior: the git-history corpus's reuse is *consecutive*-shaped, so the
  semantic uplift may be modest; the oracle decides.
- **Run on:** the same real session transcripts the prefix-cache oracle is pending on (the generator
  `tools/bench/make_git_session_corpus.py` is `--repo`-agnostic; the production number comes from
  real session logs). **No kernel needed to validate.**

### 1.7 Apple-Silicon feasibility

**Favorable — pure CPU bookkeeping in unified memory.** The hashed-n-gram embedding is integer hashing
+ a float dot product (microseconds for a few-KB sketch). The index is a few-MB `Vec` co-bounded with
the prefix cache. The reuse path is the **already-landed `restore_into`** (a memcpy into the KV
arena). No weight gather, no new dispatch, no second model on the bus. The only per-request cost is
one sketch + N cosines + one token-compare on a hit — negligible against a prefill it might skip.
The unified-memory point in bible §7.0 applies: the index shares the same bus, but it is KB–MB, not
GB, so it is free relative to weights.

---

## 2. Lever B — Online vocab / draft specialization (L3.1)

L3.1 is two mechanisms under one thesis ("the model molds to its owner"): (a) **usage-frequency
vocab pruning** of the output head, and (b) **draft tuning** on the user's accept/reject history.
They share the runtime-observation infrastructure (watch the live token distribution) but have
independent oracles and independent exactness arguments, so they are designed separately.

### 2.1a — Usage-frequency vocab prune with a certifiable exact-greedy screen

- **§8 lever:** L3.1 first clause — "prune the output head to the vocabulary actually in use."
- **Why this is a *different* mechanism than the dead SVD screen:** the lm_head SVD screen is
  **FULL-RANK NO-GO** (`svd_lmhead.json`, rank99 = 97% of dim) — there is no low-rank structure to
  exploit, so a rank-r projection cannot skip full-vocab logits. **Usage-frequency pruning does not
  touch rank.** It exploits a different, measured property: a single user's session uses a tiny
  fraction of the 151,936-token vocab as the *argmax* (the calibration corpus already showed 28.4%
  of a 102K vocab ever appears, 99.5% covered by 23,628 tokens — `vocab_prune.rs` docstring). The
  lever reads a **smaller slice of the lm_head per token** (it is ~4–10% of bytes/token), which is a
  real, if modest, axis-2 byte cut, *and* shrinks the argmax pass.
- **The exactness hole in the LANDED `PrunedVocab`, and how L3.1 closes it.** `PrunedVocab` is
  static and **not exact**: its own `argmax_round_trip_via_pruned_space` test only holds "when the
  top-1 token is in the whitelist." If the true full-vocab argmax is a pruned token, static pruning
  silently emits the wrong token — that breaks greedy-exactness. L3.1's contribution is the
  **certifiable screen** that makes it exact:

  **Mechanism (certifiable screen / safe-prune):** keep the full lm_head resident, but maintain a
  dynamic **hot set** H of the user's high-frequency argmax tokens. Each decode step:
  1. Compute logits over **H only** (the cheap pass) → candidate argmax `c` with logit `ℓ_c`.
  2. **Certify** that `c` is the true full-vocab argmax without a full pass, OR fall back. Two
     designs for the certificate, in order of preference:
     - **(Preferred) Periodic full-pass anchoring + margin:** run a *full* lm_head pass every M
       tokens (and on any low-margin step) to (a) emit exactly and (b) refresh H. Between anchors,
       trust H only when the **margin** `ℓ_c − ℓ_{2nd}` exceeds a threshold *and* H is known to
       contain the top-k from the last full pass. This is **Q-bounded between anchors, E at anchors**
       — not strictly E, so it is the *quality-bounded* variant; documented but not the default.
     - **(CHOSEN, exact) Always-include-argmax via a complete cheap pass:** the screen is exact iff
       the screened set provably contains the true argmax. The only way to *prove* that cheaply
       without low-rank structure (which is dead) is to make the "cheap" pass still **complete** but
       **cheaper-per-element** — i.e. this collapses to the existing **vocab-prune-as-a-fixed-slice
       only when the slice is guaranteed to contain the argmax.** Since we cannot guarantee that
       statically, the **exact** L3.1 vocab mechanism is: **run the full pass, but use the hot set H
       to short-circuit the *argmax reduction*, not the matmul** — no, the matmul is the cost.
  - **Resolution (the honest exact design):** a usage-frequency prune is **exact only with a
    fallback**: compute logits over H; **also** keep a cheap, exact *upper-bound certificate* that no
    out-of-H token can exceed `ℓ_c`. With no low-rank structure, the cheapest sound certificate is a
    **per-out-of-H-token norm bound**: precompute, offline, `‖w_v‖` for every vocab row and the fact
    that `ℓ_v = w_v · h ≤ ‖w_v‖ · ‖h‖`. At runtime, with `‖h‖` known, any out-of-H token whose
    `‖w_v‖·‖h‖ < ℓ_c` **provably cannot be the argmax** and is safely skipped. If *all* out-of-H
    tokens satisfy the bound → `c` is certified the exact argmax (E, no full matmul). If any
    out-of-H token's bound exceeds `ℓ_c` → **fall back to the full pass** for that token (exact). The
    hot set H is sized so the fall-back rate is low; the bound is a single multiply per skipped row
    (cheap) and the skipped rows' weights are **not read** (the byte cut).
- **Exact?** **E (greedy-lossless)** in the CHOSEN design: every emitted token is either certified
  by the norm bound to be the true argmax, or produced by a full exact pass on fall-back. The hot
  set and frequency stats change *only how often the fast path is taken*, never *which token is
  emitted*. This is the L3.1 exactness guarantee the bible names ("E if paired with a certifiable
  screen … exact greedy when the true argmax is in the screened set").
- **Energy verdict:** **GENUINE** (reading fewer lm_head rows when certified = fewer bus bytes;
  lm_head ~4–10% of bytes/token, so this is a small but real byte cut, and it does *less work*).

### 2.1b — Draft tuning on the user's accept/reject history

- **§8 lever:** L3.1 second clause — "tune the speculative draft on the user's accept/reject
  history."
- **Mechanism:** the live spec mechanism is the **n-gram / PLD / SAM draft** (τ≈1.43 on code,
  `spec_accept.json`), *not* a trained head (EAGLE-3 is NO-GO). Online specialization grows a
  **per-user suffix-automaton / n-gram index** from the user's own emitted token stream and accepted
  drafts, so the draft proposes from *this user's* repeats — their codebase identifiers, their
  boilerplate, their frequent completions — which the generic PLD-on-the-prompt draft does not see
  until they appear in-prompt. The accept/reject history tunes the draft *length* per context (the
  acc_hist in `spec_accept.json` is bimodal: 85% miss, hits run long → propose aggressively only
  where the per-context hit-rate is high) and **prunes the n-gram table to the user's
  high-acceptance grams** (drop grams that consistently mis-predict, raising mean accepted length).
- **Exact?** **E (lossless)** — unconditionally. Drafts only affect *speed*; every emitted token is
  the **verifier's** token (the landed `verify_draft_ids_until_mismatch` / `forward_tokens_verify`
  loop emits only model-verified tokens — see `qwen_dense.rs` ~1764–1812 and the
  `propose_first`/parity comments). Tuning the draft changes acceptance, never output. This is the
  safest lever in the doc: it cannot regress quality by construction.
- **Energy verdict:** **NEUTRAL-to-GENUINE** (better acceptance = fewer verify forwards per accepted
  token on favorable spans; the n-gram draft is ~free CPU automaton, bible §3 / §7.2.a).

### 2.2 The runtime-observation infrastructure (shared by 2.1a + 2.1b)

Both mechanisms need to **watch the user's live token distribution** with zero hot-path cost when
off. Build it as a **process-global, flag-gated side-observer**, copying the `attn_capture.rs`
pattern exactly:

- A `crate::stateful::usage_capture` module: `Mutex<Option<UsageState>>`, enabled by
  `DISMANTLE_QWEN_USAGE_CAPTURE` (off → every entry point is a cheap `if` that returns). Hooks at the
  point each emitted token is recorded (the `self.sampler.record(id)` / `head.note_token(id)` sites
  in `qwen_dense.rs`, e.g. ~1733, ~1784). Accumulates: per-token argmax-id frequency histogram (for
  2.1a's hot set), and the draft accept/reject ledger keyed by n-gram context (for 2.1b — the loop
  already tracks `stats.draft_accepted` / `stats.draft_rejected`). Flushes JSON to
  `$DISMANTLE_USAGE_CAPTURE_OUT`, read by the offline oracles.
- This instrument **is the L3.1 oracle's data source** and, once greenlit, the runtime adaptation's
  data source — same as `attn_capture` is both the L1.1 oracle and (would have been) the eviction
  policy's signal. Build the *observer* first (cheap, parity-neutral), run the oracle, then build the
  *adaptation* only if the oracle clears.

### 2.3 Build plan (files / APIs)

- **`crates/dismantle-core/src/stateful/usage_capture.rs`** (new) — the side-observer above. Mirror
  `attn_capture.rs` structure (Mutex, `enabled()`, `record_argmax(id)`, `record_draft(ctx, accepted,
  rejected)`, `flush()`). **Parity-neutral; lands first, behind a default-off flag.**
- **`crates/dismantle-core/src/vocab_prune.rs`** — add the **exact certifiable screen**: a
  `CertifiedVocabScreen` that holds the precomputed per-row `‖w_v‖` (offline, from the dequantized
  lm_head), a dynamic hot set `H` (updated from `usage_capture`), and a `screen(h: &[f32], h_norm:
  f32) -> ScreenResult { CertifiedArgmax(orig_id) | NeedsFullPass }`. Keep `PrunedVocab`'s mapping
  helpers; **the new type adds the soundness certificate the static `PrunedVocab` lacks.** Unit tests
  must assert the screen never disagrees with a brute-force full argmax over random `h` vectors
  (the exactness gate), and that the fall-back fires when the bound is exceeded.
- **`crates/dismantle-core/src/speculate/`** — add a `UserNgramDraft` (per-user suffix-automaton /
  n-gram index built from `usage_capture`'s emitted stream) implementing the draft-proposal shape the
  propose-verify loop expects (a `propose(ctx, k) -> Vec<u32>` analogous to
  `propose_rollout_chained`). **Reuse `verify_draft_ids_until_mismatch` unchanged** — output stays
  exact. Periodic offline-ish rebuild of the index from accumulated history (cheap, CPU).
- **`crates/dismantle-core/src/model/qwen_dense.rs`** — wire `usage_capture` record calls at the
  emit sites (behind the flag); wire `CertifiedVocabScreen` into the lm_head/argmax path behind
  `DISMANTLE_QWEN_VOCAB_SCREEN` (opt-in); wire `UserNgramDraft` as a draft source behind a flag.
  Each is an independent, small, reviewable diff.
- **`tools/bench/oracle_vocab_coverage.py`** (new) + extend **`oracle_spec_accept.py`** — see §2.4.

### 2.4 Oracle / gate to greenlight each half

- **2.1a vocab screen oracle (`oracle_vocab_coverage.py`, new — extends `oracle_svd_lmhead.py`'s
  data-loading):** on a real session token stream (from `usage_capture` or a transcript), measure
  (a) **effective vocab coverage** — what fraction of 151,936 is ever the argmax, and the hot-set
  size H that covers ≥99.x% of argmax steps; and (b) the **certificate fall-back rate** — simulate
  the norm-bound screen (precompute `‖w_v‖` from the GGUF lm_head, exactly as `oracle_svd_lmhead.py`
  dequantizes it; feed real hidden-state norms if available, else a `‖h‖` distribution) and count how
  often `max_{v∉H} ‖w_v‖·‖h‖ < ℓ_c` holds (fast path) vs requires a full pass.
  **Greenlight:** build the screen when a small H (say ≤ a few K tokens) yields a **high certified-
  fast-path rate** (e.g. ≥80% of tokens skip the bulk of the lm_head) — i.e. the byte cut is real and
  the fall-back is rare. If `‖h‖` is large enough that the norm bound rarely certifies (the bound is
  loose because weight norms are similar — plausible given the FULL-RANK, cond≈45 spectrum), the
  exact screen **dies cheaply** (record it; the byte cut isn't worth the per-row bound multiply).
  *This is the honest risk:* a full-rank head with similar row norms is exactly the regime where a
  norm-bound certificate is weak — the oracle must confirm the bound actually certifies before any
  wiring. **NumPy afternoon, no kernel.**
- **2.1b draft-tuning oracle (extend `oracle_spec_accept.py`):** the existing oracle measures generic
  PLD τ=1.43. Add a **user-specialized** variant: build the n-gram index from a *prior* slice of the
  user's stream (their history) and measure τ on a *held-out* later slice (warm-start the automaton
  with the user's repeats, not just the in-prompt suffix). **Greenlight:** adopt user-draft
  specialization if it lifts warm τ **materially above the 1.43 generic baseline** toward the 2.5
  gate (even a partial lift, say τ≥1.8, is a real per-user win since the draft is free and lossless —
  unlike a trained head, there is no net-negative risk). If specialization does not move τ (the
  user's repeats are already captured by in-prompt PLD), record NO-GO; the generic draft stays.
  **CPU, no kernel.**

### 2.5 Apple-Silicon feasibility

**Favorable for both halves.**
- **Vocab screen:** the certificate is one multiply per out-of-H row (`‖w_v‖·‖h‖`) — cheap ALU
  against the bible's >20× compute headroom (§7.0) — and the *win* is **not reading** the skipped
  rows' weights, a contiguous byte cut on the lm_head (no gather; H and the skipped set are row
  ranges / a precomputed mask). The full-pass fall-back is the existing lm_head GEMV. Compatible with
  the tied-embedding Q6_K head (the screen operates on the dequantized/predec rows the engine already
  streams).
- **Draft tuning:** the n-gram/SAM index is a **CPU automaton** (~zero GPU cost, bible §3 / §7.2.a —
  the explicitly Apple-Silicon-safe spec path) overlapped with the GPU verify. The verify is the
  K-wide GEMM the engine already shapes (`forward_tokens_verify`). No new kernel, no ANE, no second
  model. The per-user index is KB–MB in unified memory.

---

## 3. Summary, confidence ranking, and the first move

| | Lever A — Semantic cache (L1.2 ext) | Lever B-a — Vocab screen (L3.1) | Lever B-b — Draft tuning (L3.1) |
|---|---|---|---|
| §8 lever | L1.2 semantic half | L3.1 vocab prune | L3.1 draft specialization |
| mechanism | n-gram-embed near-dup → exact-common-prefix verify → reuse | usage-frequency hot set + norm-bound certificate over full lm_head | per-user n-gram/SAM index grown from accept history |
| exactness | **E** (verify = exact token-prefix compare; QualityBounded mode opt-in, Q) | **E** (norm-bound certificate or full-pass fall-back) | **E** (verifier emits; drafts affect only speed) |
| builds on (landed) | `InMemoryPrefixCache` + `restore_into` + `SemanticIndex` stub | `PrunedVocab` + lm_head/argmax path | `verify_draft_ids_until_mismatch` + propose-verify loop |
| new infra | `InMemorySemanticIndex` + hashed-n-gram sketch | `usage_capture` + `CertifiedVocabScreen` | `usage_capture` + `UserNgramDraft` |
| oracle | extend `oracle_prefix_cache.py` (incremental near-dup reuse over exact) | `oracle_vocab_coverage.py` (coverage + certificate fall-back rate) | extend `oracle_spec_accept.py` (user-warm-start τ vs 1.43) |
| greenlight | semantic uplift ≥ ~10 pts over exact-only @ ≥95% verify-confirm | small H gives ≥80% certified fast path (bound actually certifies) | warm τ materially > 1.43 toward 2.5 |
| energy | GENUINE (skipped forwards) | GENUINE (fewer lm_head rows read) | NEUTRAL-to-GENUINE (fewer verify forwards) |
| Apple-Si | favorable (CPU bookkeeping, reuses landed memcpy restore) | favorable (cheap bound multiply, contiguous byte cut, no gather) | favorable (CPU automaton + existing verify GEMM) |
| honest risk | near-dups may be rare beyond consecutive (exact tier already wins) | full-rank head + similar row norms → loose bound may rarely certify | user repeats may already be caught by in-prompt PLD |

**Higher-confidence lever: Lever B-b (draft tuning).** It is the only lever in the doc that is
**E by construction with zero quality-regression risk** (the verifier emits; tuning the draft cannot
produce a wrong token), it builds on a **fully landed, exercised** propose-verify loop, its draft is
the Apple-Silicon-safe CPU automaton the bible blesses, and it improves the *one spec mechanism still
alive* (n-gram τ=1.43) after the trained head was killed. Its worst case is "no τ lift, draft stays
generic" — a cheap NO-GO, never a regression. **Confidence: M-H.**

**Lever A (semantic cache)** is next — also E in its default mode, also pure bookkeeping atop landed
infra, but its *payoff* (incremental reuse beyond exact-consecutive) is less certain than B-b's
safety, because the exact prefix cache already captures the dominant focused-edit-loop case; the
semantic uplift is real only if non-consecutive near-duplicates are common, which the oracle must
confirm. **Confidence: M** (mechanism H, payoff M — matching the bible's "M for semantic vs H for
prefix").

**Lever B-a (vocab screen)** is the **lowest-confidence** of the three, and the doc says so plainly:
the lm_head is full-rank with condition number ~45 and similar row magnitudes, which is precisely the
regime where a norm-bound certificate is **weak** (loose upper bounds rarely certify), and the lm_head
is only ~4–10% of bytes/token so the ceiling is small even if the bound holds. It is included because
its **exactness mechanism is genuinely different from the dead SVD screen** (frequency + certificate,
not rank) and the oracle is a cheap NumPy afternoon — but the honest expectation is that it may die at
the certificate-strength oracle. **Confidence: L-M.** Do not build it before the fall-back-rate oracle
clears.

**The first move (matches bible §8.3 Phase A discipline):** build the **`usage_capture`
side-observer** (parity-neutral, default-off, the cheap `attn_capture`-shaped instrument) and run
**both L3.1 oracles** (`oracle_vocab_coverage.py` + the user-warm-start extension of
`oracle_spec_accept.py`) plus the **semantic-uplift extension of `oracle_prefix_cache.py`** — three
NumPy/CPU afternoons, no kernel, no GPU. They rank the three levers by measured payoff before a single
body is written. Then build **Lever B-b** first if its τ oracle clears (highest confidence, zero
regression risk), Lever A second if its uplift oracle clears, and Lever B-a only if its
certificate-strength oracle defies the full-rank prior.

---

*Companion docs: `plans/throughput_bible_2026_05_30.md` §8 (the System-Level Shift),
`plans/stateful_core_design_2026_05_30.md` (the L1.1/L1.2 interface scaffold),
`crates/dismantle-core/src/stateful/prefix_cache.rs` (the landed exact prefix cache + `SemanticIndex`
stub), `crates/dismantle-core/src/vocab_prune.rs` (the landed static prune this exact-screens),
`crates/dismantle-core/src/speculate/shared.rs` (the landed lossless verify), `reports/dead_levers.md`
(EAGLE-3 head + L1.1 KV working-set + SVD lm_head — the kills that bound these designs).*
