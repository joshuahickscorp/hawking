# Oracle — small DENSE-draft speculative decoding (axis-3 spec; the reframe the EAGLE-head kill does NOT touch)

**Lever:** use a small dense Qwen (0.5B or 1.5B Q4_K_M) as the draft model for the
3B Q4_K_M target, lossless speculative decoding.
**Lane:** design + CPU accept-math skeleton + threshold (this doc). Decisive
measurement is the GPU lane / Colab (two model forwards + a paired cost bench).
**Date:** 2026-05-31
**Harness:** `tools/bench/draft_accept_oracle.py` (selftest GREEN; sweep + logits
paths exercised on synthetic data).

> **Why this is a live reframe, not a re-spawn of a dead lever.** The
> `dead_levers.md` entry "EAGLE-3 trained draft head (Eagle5 v3)" is a Type-1/2
> kill of a *trained EAGLE head* (one extra transformer block conditioned on the
> target's hidden state, trained on a Q4_K_M capture; τ=0.877 on code, net-
> negative on-device). A **small dense draft is a different mechanism**: an
> independently-pretrained standalone LM that drafts tokens from its own forward
> pass — no hidden-state coupling, no training, no head↔runtime forward-parity
> gap. Its acceptance is governed by how often a 0.5B/1.5B model's next token
> matches the 3B's, which is an empirical fact nobody has measured here yet. The
> EAGLE kill says nothing about it. (It is also distinct from the n-gram/PLD
> oracle, where the draft is a free CPU automaton so τ is the speedup ceiling;
> here the draft costs real forwards, so τ must clear a breakeven.)

---

## 1. Local model inventory (present vs needs-download)

`find` under the model dir oracle_qtip_quality.py uses
(`/Users/scammermike/Downloads/dismantle/models/`) plus the HF cache:

| role | file | bytes | tensor-bytes | present? |
|---|---|---|---|---|
| **target** | `models/qwen2.5-3b-instruct-q4_k_m.gguf` | 1,929,903,264 | 1,923.9 MB | **yes** |
| draft A | `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` | 491,400,032 | 485.5 MB | **yes** |
| draft B | `models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1,117,320,736 | 1,111.4 MB | **yes** |
| (f16 1.5B for Colab) | HF cache `models--Qwen--Qwen2.5-1.5B-Instruct/.../model.safetensors` | — | — | **yes (f16)** |
| (f16 0.5B for Colab) | — | — | — | **no — download on Colab** |

**Both draft candidates are on disk as Q4_K_M** — the on-device paired bench (the
decisive cost measurement, §4) can run with zero downloads. A 1.5B **f16**
safetensors is also cached locally; the 0.5B f16 would be a Colab download. Same
tokenizer/vocab family across all three (Qwen2.5), so logits are directly
comparable position-for-position — no vocab remap needed. (Aside: `qwen2.5-7b`
and `deepseek-v2-lite` GGUFs are also on disk but are not draft candidates here.)

---

## 2. Harness contract — `tools/bench/draft_accept_oracle.py`

Three modes; the model-forward part is explicitly the GPU/Colab step and is NOT
run here.

| mode | input | runs where | output |
|---|---|---|---|
| `--selftest` | synthetic | in-session (CPU) | asserts accept math + breakeven; non-zero exit on any failure |
| `--sweep` | none | in-session (CPU) | the GO/NO-GO τ threshold tables (§3) |
| `--logits T.npy D.npy` | **GPU-lane-exported** aligned logits | in-session math on GPU output | per-k τ, accept hist, speedup vs AR + vs ngram, verdict; writes `reports/oracle/small_draft_accept.json` |

**The `--logits` contract (what the GPU lane must export):**
- `T.npy`, `D.npy` = float arrays of shape **(T, V)**, the **next-token logits**
  of the **target** and **draft** at the *same* T positions over one contiguous
  **code** token stream (position *t* predicts token *t+1*). Must be aligned and
  same shape; the harness asserts this.
- Produced by forwarding *both* models on the *same* held-out code tokens (the
  target's own greedy continuation is the natural stream to score, so the draft
  is judged on the distribution it will actually face). This forward pass is the
  GPU/Colab step — the harness never loads a model.
- `--temperature 0` (default) scores the **greedy** regime = the dismantle
  default decode: a drafted token is accepted iff `argmax(draft) == argmax(target)`
  at that position (the deterministic special case of the Leviathan rule).
  `--temperature τ>0` additionally scores the stochastic regime via the
  TV-overlap accept probability `Σ_x min(p_x, q_x)` (`--exact-accept-prob`,
  deterministic) or a Monte-Carlo draft sample.

**The accept rule (lossless, exact).** Leviathan/Chen et al. "Fast Inference from
Transformers via Speculative Decoding" (2023): draft proposes *k* tokens from its
dist *q*; the verifier accepts token *i* with prob `min(1, p_i/q_i)` (*p* =
target dist), stops at the first reject, and the target emits one correction
token from the residual `(p−q)_+`; if all *k* accept, the target emits one bonus
token from *p*. **Tokens emitted per cycle = (#accepted) + 1**, with #accepted ∈
[0, k]. **τ = mean accepted length = E[tokens emitted per verify forward]**. The
output is bit-identical to plain target sampling — so any τ>breakeven is a free,
zero-quality-risk speedup (same property the n-gram oracle relies on).

**Self-test (the in-session trust gate, all GREEN):**
- agreeing greedy stream → τ = k+1 exactly (k=4→5.0, k=8→9.0);
- disagreeing greedy stream → τ = 1.0 for all k;
- a known accept-2-then-miss pattern (`T,T,F,…`) → τ = 3.0 exactly for k≥2;
- TV-overlap accept prob endpoints: identical dists → τ=k+1, disjoint → τ=1.0;
- breakeven algebra is unity at the derived threshold, monotone in c;
- `τ(α,k)` geometric model endpoints + monotonicity.

---

## 3. The GO/NO-GO threshold (the decision math)

### 3.1 Cost model (decode is bandwidth-bound here)

Per **cycle**: the draft proposes *k* tokens **sequentially** (*k* draft forwards,
cost `k·c`), then **one** target forward (cost `1 + v`) verifies all *k* in
parallel and emits the bonus token. `c` = draft/target forward-cost ratio (target
forward = 1.0 unit); `v` = verify overhead (logit compare + KV bookkeeping; ≈0 on
a fused runtime). Yield = τ tokens. So:

```
spec rate (tokens per target-forward-time) = τ / (k·c + 1 + v)
```

**Grounding `c` (label = ESTIMATE; the on-device paired bench is decisive).**
Decode on this engine is bandwidth-bound (~85% GPU-busy; Q4_K predec GEMV at the
HW memory-model optimum — MEMORY.md). A decode step streams the whole model once,
so per-token forward cost ≈ total tensor bytes streamed. Measured from the local
GGUF headers (no dequant):

| draft | tensor-byte ratio vs 3B | `c` point | `c` bracket | why bracketed |
|---|---|---|---|---|
| 0.5B | **0.252** | 0.25 | [0.22, 0.30] | KV + per-dispatch overhead scale sub-linearly with size (favor smaller draft → lower c); GPU underfill of tiny GEMVs pushes the other way (→ higher c) |
| 1.5B | **0.578** | 0.58 | [0.52, 0.65] | same two corrections, smaller in magnitude |

The byte ratio is the honest first estimate; the **on-device paired forward-cost
bench (§4) collapses each interval** to a measured number, which `--c` plugs into
the verdict.

### 3.2 Breakeven inequalities

Baselines the dense draft must beat:
- **plain autoregressive**: rate = 1.0 tok / target-forward.
- **user-ngram "bonus-first"** (the task's stated competitor): spends **2 target
  forwards per cycle** and emits `τ_ngram` tokens, so rate = `τ_ngram / 2`. On
  *generic* code `τ_ngram = 1.43` (`reports/oracle/spec_accept.json`), giving rate
  `1.43/2 = 0.715` — i.e. **the ngram bonus-first path is itself slower than plain
  AR on generic code**, so the binding baseline a dense draft must clear is
  `max(1.0, τ_ngram/2) = 1.0` (plain AR) unless the user-specialized warm τ is
  high. (Warm-started per-user ngram pools to τ_suffix≈3.4 in
  `spec_accept_warmstart.json` → bonus-first rate ≈1.7; on such users the ngram
  bar is the binding one. The harness reports the threshold against BOTH.)

```
beats plain AR             ⇔  τ  >  k·c + 1 + v
beats ngram bonus-first    ⇔  τ  >  (τ_ngram / 2)·(k·c + 1 + v)
speedup_vs_AR  S_ar  = τ / (k·c + 1 + v)
speedup_vs_ngram     = S_ar / (τ_ngram / 2)
```

### 3.3 The thresholds for each draft (the headline numbers)

τ threshold = minimum mean-accepted-length to **win**, at v=0, point-estimate c
(`--sweep` reproduces the full table incl. the c-brackets):

| draft (c) | k | τ > X to beat **plain AR** | τ > X to beat **ngram (τ_ng=1.43)** |
|---|---|---|---|
| **0.5B** (c=0.25) | 2 | 1.50 | 1.07 |
|  | 3 | 1.75 | 1.25 |
|  | **4** | **2.00** | 1.43 |
|  | 5 | 2.25 | 1.61 |
|  | 6 | 2.50 | 1.79 |
|  | 8 | 3.00 | 2.15 |
| **1.5B** (c=0.58) | 2 | 2.16 | 1.54 |
|  | 3 | 2.74 | 1.96 |
|  | **4** | **3.32** | 2.37 |
|  | 5 | 3.90 | 2.79 |
|  | 6 | 4.48 | 3.20 |
|  | 8 | 5.64 | 4.03 |

Reading the thresholds against an achievable-τ model (geometric, constant
per-token accept rate α; `--sweep` second table):

| α | τ @ k=4 | clears 0.5B k=4 (need 2.00)? | clears 1.5B k=4 (need 3.32)? |
|---|---|---|---|
| 0.50 | 1.94 | no (just short) | no |
| 0.60 | 2.31 | **yes** | no |
| 0.70 | 2.77 | **yes** | no |
| 0.80 | 3.36 | **yes** | **yes (just)** |
| 0.90 | 4.10 | **yes** | **yes** |

**Threshold verdict (direction, NOT the gate):**
- **0.5B draft is the favorable bet.** Because c≈0.25 is small, the breakeven is
  low: a per-token accept rate **α ≈ 0.6** (τ≈2.31 at k=4) already beats plain AR,
  and α≈0.5 is right at the edge. Same-family small→large drafts on code commonly
  sit in the 0.6–0.8 greedy-agreement range *(label = ESTIMATE, literature/
  expectation — NOT measured on Qwen2.5 0.5B→3B here)*. If the real measured α
  lands ≥0.6, the 0.5B draft is a GO with margin; the optimal k is small (2–4) and
  the harness picks it automatically.
- **1.5B draft only wins if acceptance is high.** c≈0.58 pushes the k=4 breakeven
  to τ>3.32 (α≈0.8). A 1.5B model agrees with the 3B more often than the 0.5B, but
  it must agree *a lot* more to overcome ~2.3× the draft cost. This is the case the
  proxy genuinely cannot call — it hinges on whether the 0.5B→1.5B accept-rate lift
  is big enough, which **only the measured logits decide**.
- **NEEDS-MEASUREMENT, not a kill.** No CPU/weight-only proxy can produce the
  cross-model greedy agreement rate (it needs two forward passes), so per the
  Kill-Protocol this lever is **NEEDS-MEASUREMENT**, not NO-GO. The decisive gate
  is §4. (A wrong simulated α would be worse than none.)

---

## 4. The decisive GPU-lane / Colab measurement to run

Two measurements; both are needed for a verdict. Either lane works (local Metal
GPU lane preferred — both Q4_K_M drafts are already on disk).

**(A) Accept-rate measurement → the real τ.**
1. Pick a held-out **code** prompt set (reuse the corpus behind
   `reports/oracle/spec_accept.json`, ~40k code tokens; or `a4_code_prompt.txt`).
2. Run the **3B target** greedily to produce a continuation stream of T tokens;
   record the target's next-token logits at every position → `T.npy` (T, V).
3. Forward the **draft** (0.5B, then 1.5B) over the *same* token stream (teacher-
   forced on the target's tokens); record draft next-token logits → `D.npy`.
   *(Greedy regime needs only the two argmax streams; exporting full logits also
   enables the temperature>0 TV-overlap metric.)*
4. `draft_accept_oracle.py --logits T.npy D.npy --draft 0.5B` → measured τ per k,
   the accept histogram, and the speedup verdict. Repeat `--draft 1.5B`.
   - Keep V to the **pruned 32K vocab** the engine actually decodes (LM-head is
     vocab-pruned — MEMORY.md), so the logit arrays stay <1 GB and the accept rate
     matches production. Per-position argmax-only export is even cheaper.

**(B) Cost-ratio measurement → the real c (collapses the [lo,hi] bracket).**
5. Paired single-token decode bench (mirror `eagle5_paired_bench.sh` /
   `paired_lever.sh` discipline, locked env, code prompt): measure draft-only
   decode tps and target-only decode tps on this machine. `c = target_ms_per_tok /
   draft_ms_per_tok`'s reciprocal, i.e. `c = draft_forward_ms / target_forward_ms`.
   This is the number to pass as `--c` (it supersedes the byte-ratio estimate; it
   captures KV + dispatch overhead + GPU-fill the byte ratio omits).

**(C) Verdict.** Feed measured τ (from A) and measured c (from B):
`draft_accept_oracle.py --logits T.npy D.npy --draft 0.5B --c <measured>`.
- **GO** if `S_ar > 1` at some k (the harness reports best-k). Strong GO if it also
  beats the ngram bonus-first rate. Then the lever earns a real spec-decode wiring
  task (the runtime already has a spec-decode path — `spec_decode_runtime_healthy`).
- **NO-GO** only on the *measured* τ/c (then it is a Type-1 kill: "this draft's
  accept rate on code is too low to clear its own forward cost", record in
  `dead_levers.md`). Do **not** kill on the proxy.

**Pre-flight checks before spending GPU time:**
- The `--sweep` table tells the lane the **target τ** for each (draft, k) — if the
  measured per-position argmax-agreement (a 1-line numpy on the two argmax streams)
  is already below the k=2 plain-AR threshold (1.50 for 0.5B, 2.16 for 1.5B in τ
  terms ⇔ α below ~0.5 / ~0.6), the draft is dead and the full sweep is skippable.
- Confirm the draft and target share the exact tokenizer/vocab (they do — same
  Qwen2.5 family) so logits align position-for-position.

---

## 5. Honesty ledger

- **All τ thresholds = MEASURED-from-byte-counts cost × EXACT algebra** (the
  breakeven inequalities are self-test-verified). The cost ratio `c` is an
  **ESTIMATE** (byte ratio) until the §4(B) paired bench; the brackets [0.22,0.30]
  / [0.52,0.65] carry that uncertainty.
- **The acceptance rate α / τ on real code is UNMEASURED** — no CPU/weight proxy
  can produce it (needs two forward passes). The α=0.5–0.9 row is an illustrative
  *literature/expectation* band, explicitly tagged ESTIMATE, used only to read the
  thresholds — it is NOT a claimed result.
- **The single decisive gate:** §4(A) measured greedy agreement (→ real τ on code)
  + §4(B) measured paired forward-cost ratio (→ real c). With both, the verdict is
  a one-command read; without them this is direction-only.
- **Quality risk = zero** by construction: lossless speculative decoding emits the
  target's exact distribution. The only question is speed, i.e. whether τ clears
  the breakeven. There is no quality oracle to run.
- **Caveats:** (1) teacher-forcing the draft on the target's stream measures
  acceptance on the distribution the draft actually faces — correct; do not score
  the draft on its *own* free-running stream. (2) Sequential draft cost `k·c`
  assumes the *k* draft tokens are generated one-by-one (they are, in standard
  spec decode); a tree/Medusa draft changes the cost model and is out of scope
  here. (3) v (verify overhead) is taken ≈0; if the runtime's verify forward of k
  tokens costs materially more than one decode step, fold it into `--c`'s
  measurement or raise the threshold accordingly.
