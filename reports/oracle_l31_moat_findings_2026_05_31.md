# L3.1 stateful-moat oracle sweep — findings (2026-05-31)

**Method:** three offline CPU/NumPy kill-or-keep oracles, run *before* any body
(bible §8.3 / CLAUDE.md Kill Protocol). Corpus = git-history session proxy (14
sessions, real `llama-tokenize` BPE) — **ESTIMATE**; the production numbers come
from `usage_capture` logs later. Spread is reported, not just the mean.

Artifacts: `reports/oracle/spec_accept_warmstart.json` (Oracle 1, committed
`a0f7a6e`), `reports/oracle/semantic_uplift.json` (Oracle 2),
`reports/oracle/vocab_coverage.json` (Oracle 3).

## Ranking by measured payoff

### 1. Draft tuning (L3.1 B-b) — **GO** → build first (in flight)
- **τ_warm_suffix = 3.40 pooled / 2.33 median / 1.0–6.0 range** vs cold 2.51
  (**+0.89 additive specialization**), measured **only** on the post-shared-prefix
  tokens the shipped default-on prefix cache does NOT serve — so it is *not* the
  L1.2 win double-counted (τ_warm_full was 5.46; deliberately discounted).
  Generic baseline τ=1.43; per-user GO bar 1.8.
- **Complementary** to the prefix cache: pays off where a session ranges (low
  cache reuse); ~nothing in the fully-cached edit loop (2 degenerate sessions
  τ≈1.0, ~100% prefix-served). The two stateful levers cover different regimes.
- **E by construction** (the verifier emits) → zero regression risk, free CPU
  draft. Body building now (`usage_capture` + `UserNgramDraft`).

### 2. Semantic cache (L1.2 extension) — **NO-GO (Type-2, parked)**
- **+1.48 pts mean / median +0.00 / max +13.2** incremental reuse over the
  exact tier, vs the ~10-pt gate. 12 of 14 sessions = +0.00 (the default-on
  exact tier already harvests every consecutive shared prefix). Retrieval is
  precise (100% verify-confirm at τ_sem=0.80, MIN_REUSE=16) → the kill is about
  **opportunity, not recall**.
- **Type-2 (alive behind a named oracle):** the mechanism provably works — 2
  "return-to-a-prior-file" sessions hit +13.2 / +7.5. It died on the proxy's
  consecutive-edit *workload shape*, not on reality. **Reframe:** re-run the same
  oracle (`oracle_prefix_cache.py`, built, ~13 s, no GPU) on **real
  file-interleaved session logs**. Parked there; not resurrected on vibes.

### 3. Vocab screen (L3.1 B-a) — **NO-GO (Type-1, dead)**
- **0% certified fast-path** across the entire sweep. The norm-bound certificate
  needs `cos(w_c,h) > 1.0–1.46` (unreachable; cos ≤ 1), and does not move as H
  grows 256→32768. Coverage is fine (H=7,119 covers 99.9% of occurrences) — not
  the blocker.
- **Smoking gun:** the 10 highest-norm lm_head rows are **rare** tokens (corpus
  freq 0; freq-rank 22k–146k), so a frequency-chosen hot set never includes them
  → `max_{v∉H}‖w_v‖` is pinned at the global max → the Cauchy–Schwarz bound is
  structurally too loose. **Type-1:** a measured head property (similar row norms
  + norm/frequency anti-correlation), matching the full-rank cond≈45 prior. Named
  reframes (block-max / per-coordinate certificate; data-aware real-argmax hot
  set) recorded **dead-until-their-cheap-oracle**. lm_head is only ~4–10% of
  bytes/token → small ceiling regardless.

## Recommendation
Build **draft tuning** (in flight). Do **not** build semantic (re-gate on real
session logs first) or vocab (Type-1 dead). **1 of 3 levers live** — a correct,
common Kill-Protocol outcome, not a padded ledger. Canonical kill entries land in
`reports/dead_levers.md` at integration.
