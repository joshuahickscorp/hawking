> **STATUS: DESIGN-AHEAD — reviewable interfaces, not a wired implementation.**
> Companion to `plans/throughput_bible_2026_05_30.md` §8 ("The System-Level Shift").
> Scope: the runtime interfaces for the two strongest stateful-core levers, **L1.2
> (cross-prompt computation reuse — prefix + semantic caching)** and **L1.1 (KV cache
> as a living working set)**. This doc + the `stateful/` stub module are the deliverable;
> the bodies are `todo!()` until each lever's oracle clears (§8.3 Phase A).
> This is slightly ahead of the project's oracle-first discipline (interfaces before the
> oracle greenlights the bodies) — intentional and approved. Keep it to interfaces and
> doc comments, never behavior.

---

# `dismantle` — Stateful Core: L1.1 + L1.2 runtime interfaces

*The bible's §8 thesis: incumbents are **general, stateless, static** by requirement; dismantle's
shift is to refuse all three on one machine for one user. This doc designs the two Layer-1
levers that attack the von Neumann tax directly — reuse computation across prompts (L1.2) and
treat the KV cache as a bounded living working set rather than a linearly-growing blob (L1.1).
Both are mostly **E (bit-identical greedy)** in their safe modes, both are favorable on the M3 GPU
(bookkeeping over existing arenas; no weight gather), and both are gated by an offline oracle that
must clear before any body is implemented.*

---

## 0. What already exists (do not rebuild)

The on-disk prefix KV cache is **already shipped** and wired into the decode entry point. The new
`stateful/` module is the in-RAM, session-scoped tier that sits *in front of* it — it does not
replace it.

| existing artifact | file | role |
|---|---|---|
| `KvCache` | `crates/dismantle-core/src/cache/mod.rs` | per-layer flat fp32 `keys`/`values` Vecs + `seq_len`; `append`/`reset`/`keys_for`/`values_for`. KV is layer-synchronous (one `seq_len` across all layers). |
| `PrefillDiskCache` | `crates/dismantle-core/src/cache/prefill_disk.rs` | on-disk, cross-*session* prefix cache. Rolling-sha256 prefix-hash key, `lookup_longest_prefix`, `store`/`store_raw`, LRU-by-mtime eviction, mmap'd hits. |
| `PrefillKey` | same | `{model_hash, tokenizer_hash, prefix_hash, prompt_tokens}`; `from_model_and_prompt(model_id, tokenizer_sig, tokens)` + `rolling_prefix_hash` (the property that makes longest-prefix lookup O(N) hash-updates). |
| `PrefillHit` / `restore_hit_into_kv` | same | mmap'd hit → `restore_hit_into_kv(&hit, &mut kv)` copies K/V back into a `KvCache` and sets `seq_len`. |
| decode entry point | `crates/dismantle-core/src/model/qwen_dense.rs` `QwenDense::generate` (line ~1227) | the request entry. Prefix-cache lookup hook is **lines ~1292–1319**; store hook is **lines ~1388–1394**. Keyed on `self.model_id` + `tokenizer_signature(&self.tokenizer)` + `prompt_ids`. |

**The relationship between the existing disk cache and this design.** `PrefillDiskCache` is the
*persistence* tier: it survives process restarts, pays a disk-read + f32-copy on every hit, and is
keyed by a full rolling hash. The `stateful::prefix_cache` tier designed here is the *in-RAM,
hot-session* tier: it retains live KV blocks the decode loop just produced so the **next request in
the same session** reuses them with zero disk I/O and (eventually) zero copy. The disk cache is the
cold-start backstop behind it; the RAM cache is the thing that makes "reuse the redundant 70–90% of
a coding session" (§8.1 L1.2) actually fast. The two share the *key derivation* (`PrefillKey`'s
rolling prefix hash) so a block is addressable identically in both tiers.

---

## 1. L1.2 — Prefix cache (cross-prompt computation reuse)

### 1.1 Lever mapping

- **§8 lever:** L1.2 — cross-prompt computation reuse (prefix + semantic caching).
- **Mechanism (bible):** coding workloads re-send the same files, imports, scaffolding. Cache the
  KV for shared **prefixes** so an unchanged prefix is never recomputed. A single-user local engine
  is the *ideal* case (one tenant, one observable workload).
- **Benefit:** eliminates whole forward passes on the redundant majority of a session. **Energy
  verdict GENUINE** — a skipped forward pass is power *not drawn*, the strongest true energy win in
  §8 because it removes work entirely rather than moving it faster.
- **Exact?** **E** for prefix caching — a matched prefix is **bit-identical reuse**. See §1.4.

### 1.2 The interface

The trait `PrefixCache` abstracts a session-scoped KV-block store. Implementations:
- `InMemoryPrefixCache` — the hot tier (designed here): bounded RAM, retains live blocks.
- (future) a thin adapter over `PrefillDiskCache` so the same trait fronts the disk tier.

```
trait PrefixCache {
    fn lookup(&self, key: &PrefixKey) -> Option<PrefixMatch>;
    fn insert(&mut self, key: PrefixKey, blocks: KvBlockRange) -> InsertOutcome;
    fn evict_to(&mut self, budget: PrefixCacheBudget);
    fn stats(&self) -> PrefixCacheStats;
}
```

- **`PrefixKey`** — wraps the existing rolling-prefix-hash derivation. Carries `model_hash`,
  `tokenizer_hash`, the `prefix_hash` (rolling sha256, recoverable per-length so longest-prefix is
  cheap), and `n_tokens`. Built from the same `(model_id, tokenizer_signature, prompt_tokens)` tuple
  the decode entry point already computes — so the RAM and disk tiers are **key-compatible**.
- **`lookup`** — returns the **longest** cached prefix that is a strict prefix of the query tokens
  (strict: never the entire prompt, mirroring the disk cache's "bail one token short" rule at
  `prefill_disk.rs` so the decode loop always has a real `last_id`). A `PrefixMatch` names the
  matched length and a handle to the retained `KvBlockRange`.
- **`insert`** — after a clean prefill/decode, retain the produced KV blocks under the full-prompt
  key so a later request that *extends* this one hits. Returns `InsertOutcome` (Inserted / Replaced /
  RejectedOverBudget) — never silently drops.
- **`evict_to`** — enforce a `PrefixCacheBudget` (bytes and/or entry count). LRU by last-hit, same
  policy class as the disk tier's mtime LRU.

### 1.3 Where it hooks into the decode entry point

The hook points already exist for the disk tier; the RAM tier slots in at the **same two seams**,
in front of the disk lookup:

```
QwenDense::generate  (qwen_dense.rs ~1227)
  ├─ prompt_ids = tokenizer.encode(prompt)
  ├─ kv.reset()
  ├─ [LOOKUP SEAM ~1292–1319]
  │     key = PrefixKey::from(model_id, tokenizer_sig, prompt_ids)
  │     1. RAM:  prefix_cache.lookup(&key)              ← NEW (this design)
  │     2. else  disk.lookup_longest_prefix(...)        ← existing
  │     on hit → restore blocks into self.kv, set prefill_skipped = match.n_tokens
  ├─ prefill loop over prompt_ids[prefill_skipped..]
  ├─ [INSERT SEAM ~1388–1394]
  │     prefix_cache.insert(key, kv_block_range_of(self.kv))   ← NEW (this design)
  │     disk.store(key, &self.kv)                              ← existing
  └─ decode loop ...
```

**Critical:** this design does **not** modify `qwen_dense.rs`. The seams above describe where a
*future* wiring task would call the trait. The deliverable is the trait + stub so that wiring is a
small, reviewable diff later — exactly the two call sites the disk cache already uses.

### 1.4 The exact-match guarantee (greedy-lossless)

A prefix-cache hit is **bit-identical reuse**, not an approximation. The argument:

1. The KV state for tokens `[0..n)` is a pure function of `(model weights, tokenizer, tokens[0..n))`.
   Decode is causal: position `i`'s K/V depend only on tokens `≤ i`.
2. `PrefixKey` binds **all three**: `model_hash` (weights), `tokenizer_hash` (tokenization), and the
   rolling `prefix_hash` over the exact token ids. A collision would require a sha256 collision.
3. Therefore a key match ⇒ the cached K/V blocks are *the same bytes* a cold prefill of `tokens[0..n)`
   would produce. Reusing them and prefilling only `tokens[n..]` yields **identical logits at every
   subsequent position** ⇒ identical greedy argmax ⇒ identical output. This is **E**, no tolerance.
4. The existing disk tier already relies on this guarantee and its tests assert byte-equality
   (`prefill_disk.rs` `round_trip_basic`, `restore_round_trip`). The RAM tier inherits it because it
   uses the same key and stores the same K/V bytes.

The only correctness obligations the implementation must honor (encoded as doc-comment invariants on
the stubs): (a) never return a match longer than `prompt_len − 1`; (b) invalidate on any
`model_hash`/`tokenizer_hash` change (the disk tier does this — see `tokenizer_change_invalidates`);
(c) blocks are immutable once inserted (a later session must not mutate a shared prefix's K/V).

### 1.5 Semantic-cache extension (a later layer, noted not designed)

Past exact-match: embed recent contexts and recognize "I have computed something *near*-identical
before → reuse," with an **exact-match verification step before trusting a near-hit**. This is **Q**
for *acceptance* unless gated by the verify (then **E** again). The interface reserves space for it
via a `SemanticIndex` trait sketch (embedding-similarity probe → candidate `PrefixKey`s → verify),
but the bodies and the embedding model are **out of scope for this design** — they are a Phase-B+
follow-on (§8.1 L1.2 confidence M for semantic vs H for prefix). The stub carries the trait shape
and a `// SEMANTIC LAYER (later)` marker so the seam is visible without committing to it.

### 1.6 Oracle — what must show green to build the bodies

- **Oracle name:** the **prefix-cache hit-rate oracle** (§8.3 Phase A item 1 — "likely the
  highest-ROI, lowest-risk win; measure first"). It is **being measured separately** (a transcript
  replay, not part of this design), in the family of the existing offline oracles
  (`tools/bench/oracle_spec_accept.py`, `tools/bench/oracle_svd_lmhead.py`).
- **What it measures:** on real coding-session transcripts, replay consecutive requests and compute
  (a) the **average shared-prefix length across consecutive requests** (the direct hit-rate / how
  many forward-pass tokens a prefix hit would skip), and (b) the **near-duplicate rate** of contexts
  under an embedding-similarity threshold (the semantic-cache *potential*, for §1.5).
- **Greenlight threshold:** implement the L1.2 bodies when the oracle shows a **high shared-prefix
  hit rate on code** — i.e. a large fraction of consecutive requests share a long prefix, so the
  retained-KV reuse removes a meaningful share of total prefill forward passes per session. The
  bible expects this is high for code (re-sent files/imports/scaffolding) but it is **not assumed** —
  no kernel/body until the transcript number confirms it. The semantic layer (§1.5) is gated
  separately on the near-duplicate rate clearing a useful threshold *and* a verify keeping it E.
- **No kernel needed to validate** (bible): the oracle is pure transcript bookkeeping. This is why
  L1.2 jumps the queue — cheapest, highest-confidence, most-differentiated.

---

## 2. L1.1 — KV working set (KV cache as a living working set)

### 2.1 Lever mapping

- **§8 lever:** L1.1 — KV cache as a living working set.
- **Mechanism (bible):** attention is sparse — most cached tokens contribute almost nothing to the
  current token. Rank cached tokens by importance and **evict or compress** the low-value ones,
  keeping the KV at a **bounded working-set size** instead of a linearly-growing blob.
- **Benefit:** the single biggest capability unlock for a coding tool — whole-file/whole-codebase
  context from choking at ~32K to running at 200K+ in the same 18 GB, and cuts KV-read bandwidth at
  long context. **Energy verdict GENUINE at long context** (fewer KV bytes moved), NEUTRAL at short.
- **Exact?** **Q** (approximate attention; bounded, tunable by working-set size) — **with a lossless
  mode escape hatch** (keep all, no eviction) always available for correctness-critical runs.

### 2.2 The eviction-policy trait

The core abstraction is a **policy trait** that, given the per-token attention signal and the current
working set, decides which KV positions to retain. It abstracts the three documented policies so they
are swappable behind one interface:

```
trait KvEvictionPolicy {
    fn name(&self) -> &'static str;
    fn on_admit(&mut self, pos: usize, layer: usize);
    fn observe_attention(&mut self, layer: usize, query_pos: usize, scores: &AttentionScores);
    fn select_evictions(&mut self, layer: usize, ctx: &WorkingSetCtx) -> EvictionPlan;
    fn is_lossless(&self) -> bool { false }
}
```

Implementations (signatures stubbed, bodies `todo!()`):

| policy | what it keeps | maps to |
|---|---|---|
| `StreamingLlmPolicy` | the first *s* **attention-sink** positions + a recent window; evict the middle. | StreamingLLM |
| `H2OPolicy` | the recent window + **heavy hitters** by *cumulative attention mass* (running per-position score sum). | H2O |
| `SnapKvPolicy` | recent window + positions selected by **pooled importance** over a recent observation window (max/avg-pool the attention map, keep top-k). | SnapKV |
| `LosslessPolicy` | everything (`select_evictions` returns empty, `is_lossless() == true`). | the escape hatch |

- **`observe_attention`** is how a policy accumulates its statistic (H2O's cumulative sum, SnapKV's
  pooled window). StreamingLLM ignores it (positional rule only). Lossless ignores it.
- **`select_evictions`** returns an `EvictionPlan` — a set of KV positions to drop (and, later, a set
  to *compress* in place via the existing quant codecs rather than drop). It never proposes evicting
  beyond the bounded budget's overflow, and never evicts protected sinks/recent-window positions.

### 2.3 The bounded working set

`KvWorkingSet` owns the budget and drives the policy. It wraps (does not replace) the live `KvCache`:

```
struct KvWorkingSet<P: KvEvictionPolicy> {
    policy: P,
    budget: WorkingSetBudget,   // max retained positions (per layer)
    mode: WorkingSetMode,       // Bounded | Lossless
    // ... retained-position bookkeeping ...
}
```

- **`WorkingSetBudget`** — a bounded **retained-position count** (the working-set size). When
  admitting a new token would exceed it, the policy's `select_evictions` runs and the plan is applied
  *before* the append, so steady-state RAM/KV-read cost is `O(budget)`, not `O(seq_len)`.
- **`mode`** — `Bounded` (eviction active) or **`Lossless`** (the escape hatch: keep all, no
  eviction, identical to today's behavior; for correctness-critical/greedy-exact runs). `Lossless`
  mode forces the `LosslessPolicy` semantics regardless of the configured policy.
- **Hook (described, not wired):** the working set sits at the KV append site
  (`qwen_dense.rs` `forward_token` ~line 2316, where `kv_off = kv.seq_len * stride` and `seq_len`
  bumps after the layer loop). A future wiring would (a) feed attention scores to
  `observe_attention` during the MHA compute and (b) apply the `EvictionPlan` to compact the
  `KvCache` arenas. **This design does not touch that path.**

### 2.4 Coupling to the future fused quantized-KV attention kernel

Eviction is bookkeeping (drop/relocate KV positions) and is GPU-favorable as-is. **Compression**
(the second half of L1.1) reuses the existing quant codecs on KV tensors and is the natural pair for
the **fused quantized-KV attention kernel** (the mlx-qsdpa pattern: read 4/8-bit KV inline in one
dispatch, no FP16 buffer — bible §2 / §7.1). The trait reserves this via an `EvictionPlan` variant
for *compress-in-place* (vs *drop*) and a `KvCompressionCodec` association point, both marked
`// FUSED-QKV KERNEL COUPLING (later)`. No kernel is designed here; the interface just makes the
compress path expressible so wiring the kernel later does not reshape the trait.

### 2.5 Oracle — what must show green to build the bodies

- **Oracle name:** the **attention-mass concentration** oracle (§8.3 Phase A item 2).
- **What it measures (bible):** on a real long-context coding capture, replay attention and measure,
  **per layer, what fraction of cached tokens receive ≥99% of cumulative attention mass.** Also
  quantify the **context-length-vs-quality curve** as the working-set budget shrinks. This is the
  StreamingLLM/H2O finding **re-measured on Qwen2.5-3B and the actual workload** — not assumed.
- **Greenlight threshold:** implement the L1.1 bodies when the oracle shows that a **small bounded
  set of positions dominates the attention mass** (the documented finding — a few sinks + a recent
  window + a thin tail of heavy hitters capture ≥99%), so a bounded working set is safe at a quality
  loss the budget can tune. If attention mass is *not* concentrated on Qwen (mass spread broadly →
  any bounded budget drops load-bearing context), the lever dies on the oracle — same discipline that
  killed block-256 FFN sparsity (`plans/...` / MEMORY: "FFN block-256 sparsity DEAD"). The
  `LosslessPolicy` ships regardless (it is the no-op escape hatch and needs no oracle).

---

## 3. Summary table

| | L1.2 prefix cache | L1.1 KV working set |
|---|---|---|
| §8 lever | L1.2 cross-prompt reuse | L1.1 living working set |
| exactness | **E** (matched prefix bit-identical); semantic layer Q-unless-verified | **Q** bounded; **Lossless mode = E** escape hatch |
| core interface | `PrefixCache { lookup / insert / evict_to / stats }`, keyed by `PrefixKey` (rolling prefix hash) | `KvEvictionPolicy { observe_attention / select_evictions }` + `KvWorkingSet<P>` with bounded budget + Lossless mode |
| hook point | `generate` lookup seam ~1292 + insert seam ~1388 (in front of disk tier) | KV append site in `forward_token` ~2316 |
| oracle | **prefix-cache hit-rate** (transcript replay, measured separately) | **attention-mass concentration** (long-context capture replay) |
| greenlight | high shared-prefix hit rate on real code transcripts (removes a meaningful share of prefill forwards) | a small bounded position set captures ≥99% attention mass per layer on Qwen + tolerable quality-vs-budget curve |
| energy | GENUINE (skipped forwards) | GENUINE at long ctx, NEUTRAL short |
| kernel coupling | none (bookkeeping) | future **fused quantized-KV attention** (compress path) |
| confidence (bible) | H (prefix), M (semantic) | H |

**The first move (bible §8.3):** the **prefix-cache hit-rate oracle** on real coding transcripts.
If it returns a high hit rate — expected for code — L1.2 jumps the queue ahead of everything, and
these interfaces are what the implementing diff fills in.
