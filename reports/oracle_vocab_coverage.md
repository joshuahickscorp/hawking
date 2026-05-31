# L3.1 oracle — usage-frequency vocab screen + norm-bound certificate

**VERDICT: NO-GO — Type-1.** The norm-bound certificate cannot certify a small
frequency-hot vocab set at any physically-reachable hidden-state alignment, on a
measured property of the lm_head: the highest-norm rows are *rare* tokens, so they
never enter a frequency-chosen hot set and pin the bound at the global max.

**The deciding numbers (tied `token_embd.weight`, Q6_K, vocab 151936, hidden 2048):**
- effective vocab coverage (LABEL ESTIMATE, input-token proxy): **4.97%** of vocab
  ever appears (7,553 / 151,936 distinct over 434,032 occurrences); **H = 5,585**
  covers 99.0%, **H = 7,119** covers 99.9%.
- lm_head row norms are **tight**: median 1.113, p99 1.348, max 1.626 —
  max/median spread only **1.46x** (similar row norms, as the SVD oracle implied).
- certificate cos-threshold to certify, **optimistic** (‖w_c‖ = max row norm) =
  **1.000** at *every* H ∈ {256 … 32768}; **realistic** (‖w_c‖ = median) = **1.461**.
  Both are unreachable (cos ≤ 1). **Certified fast-path rate = 0%** across the whole
  sweep.
- **smoking gun:** the 10 highest-norm rows have corpus frequency **0** (frequency-rank
  22k–146k of 7,553 distinct seen). They are rare tokens, so `max_{v∉H}‖w_v‖` stays
  at the global max (1.626) for any frequency-chosen H up to ~66k.

- Oracle: `tools/bench/oracle_vocab_coverage.py` → `reports/oracle/vocab_coverage.json`
- Model: `models/qwen2.5-3b-instruct-q4_k_m.gguf` | Corpus: `/tmp/git_sessions_all/*.jsonl`
  (89 documents, all tokenized OK with `llama-tokenize --ids`)
- Greenlight rule: GO iff a small H (≤ 4096) certifies ≥ 80% of steps at a
  physically-reachable alignment (optimistic cos-threshold < 0.9).

## The mechanism under test — and why it is NOT the dead SVD screen

The lm_head SVD low-rank screen is **FULL-RANK NO-GO** (`reports/oracle/svd_lmhead.json`:
rank99 = 1987/2048 = 97% of dim, cond ≈ 45). This oracle tests a **different**
mechanism that does not touch rank:

1. keep a usage-frequency **hot set** H; compute logits over H only → candidate
   argmax `c` with logit `ℓ_c`;
2. an out-of-H token `v` is **provably** not the argmax iff `‖w_v‖·‖h‖ < ℓ_c`
   (Cauchy-Schwarz: `ℓ_v = w_v·h ≤ ‖w_v‖·‖h‖`);
3. if **all** out-of-H tokens satisfy the bound, `c` is the certified exact argmax —
   skip the bulk of the lm_head (the byte cut). Else fall back to a full exact pass.

This is exact-greedy by construction (every emitted token is certified the true argmax
or produced by a full pass). The question the oracle answers is purely **how often the
fast path is taken** — the certified-fast-path rate vs the fall-back rate.

## (a) Effective vocab coverage — LABEL ESTIMATE (input-token proxy)

Tokenizing the 89-document session corpus with `llama-tokenize` and using token-ID
frequency as a proxy for argmax usage (real argmax frequency needs a GPU decode we do
not run):

| metric | value |
|---|---:|
| distinct tokens ever seen | 7,553 / 151,936 (**4.97%**) |
| total token occurrences | 434,032 |
| H for 99.0% coverage | 5,585 |
| H for 99.5% coverage | 6,186 |
| H for 99.9% coverage | 7,119 |
| H for 99.99% coverage | 7,510 |

So a small hot set (a few K) **does** cover the corpus's token usage — coverage is not
the blocker. The blocker is the certificate.

## (b) Norm-bound certificate — LABEL ESTIMATE (‖h‖, ℓ_c modeled/swept)

The certificate fires for a step iff `max_{v∉H}‖w_v‖·‖h‖ < ℓ_c`. Dividing by `‖h‖`:
the screen certifies iff `ℓ_c/‖h‖ > max_{v∉H}‖w_v‖`. Since
`ℓ_c = ‖w_c‖·‖h‖·cos(w_c,h)`, the condition is **scale-free**:

> certify iff `cos(w_c,h) > max_{v∉H}‖w_v‖ / ‖w_c‖`.

`cos ∈ [-1,1]`; an argmax winner has `cos > 0`. We model `‖w_c‖` two ways (the winning
row's norm is unknown without a GPU run): **optimistic** = global max row norm (loosest
threshold, best case for the lever); **realistic** = median row norm (the winner is a
frequent, ordinary-norm token).

**lm_head row-norm distribution (the regime that decides it):**

| p0 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 | p100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.380 | 0.405 | 0.823 | 1.039 | 1.113 | 1.179 | 1.283 | 1.348 | 1.412 | 1.626 |

**cos-threshold to certify, by frequency-chosen H** (cos ≤ 1, so `reach=0` means the
certificate can never fire):

| H | max out-of-H ‖w_v‖ | cos\* optimistic (‖w_c‖=max) | reachable? | cos\* realistic (‖w_c‖=median) | reachable? |
|---:|---:|---:|:--:|---:|:--:|
| 256 | 1.626 | 1.000 | no | 1.461 | no |
| 512 | 1.626 | 1.000 | no | 1.461 | no |
| 1024 | 1.626 | 1.000 | no | 1.461 | no |
| 2048 | 1.626 | 1.000 | no | 1.461 | no |
| 4096 | 1.626 | 1.000 | no | 1.461 | no |
| 8192 | 1.626 | 1.000 | no | 1.461 | no |
| 16384 | 1.626 | 1.000 | no | 1.461 | no |
| 32768 | 1.626 | 1.000 | no | 1.461 | no |

The threshold does **not move with H** — because the max out-of-H norm stays at the
global max. **Certified-step rate = 0% over the entire `cos ∈ [0.30 … 1.00]` sweep, at
every H, in both scenarios.** The certificate never fires.

## Why the threshold is pinned — the norm/frequency disjointness

| rank by norm | token id | row norm | frequency-rank (0 = most frequent) | corpus count |
|---:|---:|---:|---:|---:|
| 1 | 38301 | 1.626 | 66,461 | 0 |
| 2 | 18826 | 1.624 | 48,075 | 0 |
| 3 | 97322 | 1.577 | 146,372 | 0 |
| 4 | 111889 | 1.557 | 138,357 | 0 |
| 5 | 79176 | 1.524 | 124,110 | 0 |

The highest-norm rows are **rare** tokens (frequency 0 in the corpus; rank tens of
thousands out of only 7,553 distinct tokens seen). A hot set chosen by **frequency**
therefore never contains them, so `max_{v∉H}‖w_v‖` is stuck at the global max (1.626)
even for H in the tens of thousands. The certificate needs the winning logit to clear a
ceiling set by a *high-norm rare row the screen can never include* — and since
`ℓ_c/‖h‖ = ‖w_c‖·cos ≤ ‖w_c‖ ≤ 1.626 = the ceiling`, the only way to clear it is
`cos(w_c,h) = 1` **and** `w_c` itself being the single max-norm row. Generic decode
hidden states do not do this.

## Kill Protocol classification

**Type-1.** The kill is a measured property of reality, independent of implementation
cleverness:
- the lm_head has **similar row norms** (max/median spread 1.46x), so a Cauchy-Schwarz
  row-norm upper bound is intrinsically loose — every out-of-H row's bound `‖w_v‖·‖h‖`
  is within ~1.5x of the winner's, far closer than the margin `ℓ_c` a real logit
  provides; and
- the **high-norm rows are anti-correlated with frequency**, so the frequency-chosen
  screen can never demote the ceiling.

Both are facts about *this head + this usage distribution*, not about how the screen is
coded. This is exactly the regime the honest prior named: a full-rank, cond≈45,
similar-row-norm head is where a norm-bound certificate is weakest. No implementation
fixes it; **the lever stays dead.** Per the contract, a recorded Type-1 is not
re-tested.

### Type-2 reframe considered (and why it needs its own oracle to revive)

A *tighter exact certificate* that does not use the scalar row-norm bound is a genuine
reframe, named here so it is not lost — but **dead until its named oracle clears**, not
resurrected on vibes:

1. **Per-coordinate / block-max bound.** Instead of one scalar `‖w_v‖`, store per
   contiguous row-block the per-coordinate max `max_j |w_v[j]|` and bound
   `ℓ_v ≤ Σ_j max_j|w_v[j]|·|h_j|`. This can be much tighter than the global row-norm
   bound **iff** `h` has a few dominant coordinates. **Cheap oracle:** recompute this
   sweep with the block-max ceiling — a reshape + max over the *same* dequantized `W`
   already loaded here, plus a modeled sparse-`h` distribution; a few lines added to
   `oracle_vocab_coverage.py`. On a similar-row-norm / cond≈45 head the gain is
   uncertain and must be measured before any wiring.
2. **Data-aware (real-argmax) hot set.** Order the vocab by **argmax** frequency from a
   real decode (the `usage_capture` side-observer the design proposes) rather than by
   input-token frequency. If the true-argmax-frequent set happens to overlap the
   high-norm rows, H could demote the ceiling. **Cheap-ish oracle:** re-run coverage
   with real argmax ids from `usage_capture`, then re-run this certificate sweep — but
   that needs a GPU decode to produce argmax ids, so it is **out of scope for this CPU
   oracle** and deferred to that instrument. (Note: it only helps if argmax-frequency
   and row-norm correlate, which the input-token disjointness above gives no reason to
   expect.)

Neither reframe is "alive" today: #1 awaits the block-max NumPy oracle; #2 awaits
`usage_capture`. The scalar-norm-bound form tested here is **Type-1 dead**.

## Honest caveats

- **LABEL ESTIMATE (a):** input-token frequency is a *proxy* for argmax/output
  frequency. Real argmax usage needs a GPU decode (not run). The coverage number
  (4.97%, H≈7K for 99.9%) is an estimate; the certificate kill, however, does **not**
  depend on it — even a perfect-coverage hot set fails because the *ceiling* is set by
  rare high-norm rows, not by coverage.
- **LABEL ESTIMATE (b):** `‖h‖` and `ℓ_c` are modeled via the `cos(w_c,h)` sweep, not
  GPU-measured. The sweep is scale-free (depends only on the norm ratio), so the
  conclusion is robust to the absolute hidden-state magnitude; it would change only if
  real `cos(w_c,h)` exceeded the per-H thresholds — which are ≥ 1.0 (impossible).
- **Ceiling is small anyway:** the lm_head is ~4–10% of bytes/token, so even a GO would
  be a modest byte cut. This NO-GO removes a low-ceiling lever cheaply.
- Row norms are taken from the **dequantized Q6_K** tied `token_embd.weight` (the only
  head available offline). An f16 head would have marginally different norms but the
  same similar-norm / norm-vs-frequency-disjoint structure; the Type-1 kill is
  structural, not a quantization artifact.
