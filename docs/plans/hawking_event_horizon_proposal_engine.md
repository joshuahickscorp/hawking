# Hawking Event Horizon — Unified Speculative Proposal Engine

**Date:** 2026-06-21 · **Status:** design (grounded in a verified literature pass + the in-tree spec-decode scaffolding + the kill ledger)

> The target model is the black hole; the proposal engine predicts what falls in next.
> Every shortcut is allowed because the target remains the law: if the guess is right, time
> collapses; if it's wrong, **quality is untouched.**

## 0. Thesis

Hawking is not "a faster sampler." It is a **runtime-native proposal market**: many cheap
predictors compete every token, **one exact target verifier** keeps output lossless, and a
**router** learns which proposer to trust by context, model family, entropy, hardware
pressure, and recent acceptance.

```
target context ─► ProposalRouter ─► candidate line / tree / span ─► exact target verifier
                       │                                              (accept longest valid
   model-free ────────┤                                               prefix/tree path; emit
   retrieval ─────────┤                                               target correction token)
   neural (per-vocab)─┤
   cross-tokenizer ───┤
   policy model ──────┘  (predicts accept, draft length, tree width, proposer budget)
```

**Losslessness is structural, not earned.** The exact verifier accepts/resamples every
drafted token via standard speculative sampling, so the output distribution is provably
unchanged *regardless of how approximate or aggressive the proposer is*
([EAGLE 2401.15077], [EAGLE-2 2406.16858], [SpecInfer 2305.09781]). This is the one invariant
that lets the router compose, swap, and gate proposers with **zero quality regression** — the
entire design rests on it.

**The defensible bleeding edge is the router, not any one proposer.** Every proposer below is
published. What no one ships is a unified, Metal-aware proposal market that optimizes
**tokens-accepted per wall-clock second** (not raw acceptance) and enables/disables each
proposer from live telemetry. That is the artifact.

## 1. Verified frontier (June 2026) — and the honest caveat

A claim-verified survey (24/25 claims confirmed; full record:
`artifacts/.../w8byf352t.output`). **Pervasive caveat: nearly every headline number is
CUDA/server/small-batch. The architectures transfer to Metal; the magnitudes do not and must
be re-measured locally.**

| method | what it buys | what it costs | tokenizer | training | Apple-Silicon risk | router slot |
|---|---|---|---|---|---|---|
| **n-gram / prompt-lookup / suffix** | exact long-span copy (code/JSON/logs/agent loops); τ=1.43 measured, **beats our trained head** | ~0 compute; table memory | **none — native to target's own tokens** | **none** | none (it's the safe base) | universal base proposer |
| **REST** [2311.08252] | retrieval-of-continuations | datastore build + lookup | none (target tokens) | none | datastore residency in unified mem | retrieval slot |
| **EAGLE-1/2** [2401.15077,2406.16858] | ~3×→~4× on 13B (single-batch) | trained head + **2nd-top-layer hidden tap**; dynamic tree bookkeeping | **shared vocab** | trained head/target | hidden-state tap into Metal fwd; dynamic-tree KV/mask cost | high-accept neural slot + **dynamic-tree control knob** |
| **EAGLE-3** [2503.01840] | up to ~5.6–6.5×; **shrinks to 1.38× @batch64 ⇒ best in our low-batch regime** | trained head + **3-layer (low/mid/high) tap**; heavier coupling | shared vocab | trained head/target | expose 3 layer activations from Metal fwd | top-accept neural slot |
| **P-EAGLE** [2602.01469] | parallel K-token draft, +1.10–1.36× over AR EAGLE-3 (server, low-concurrency) | trained **parallel** head + hidden tap | shared vocab | trained | removes per-draft launch/sync (good on Metal); gains shrink at high concurrency | parallel neural slot |
| **DFlash / DDTree** [2602.06036,2604.12989] | whole block in **one** forward (kills AR drafter overhead); DDTree = best-first-heap tree, single-pass verify w/ **ancestor-only mask** | trained block-diffusion drafter | shared vocab | trained | **DDTree is the portable Metal tree-verify blueprint** | block/tree neural slot. ⚠️ DFlash "6×/2.5×-over-EAGLE3" **REFUTED 0-3** — mechanism real, magnitude unproven |
| **OmniDraft** [2507.02659] | one drafter, many targets; up to ~1.5–1.7× cross-vocab | online n-gram cache + distillation (not training-free) | **cross-vocab** | online adapt | server-measured | heterogeneous neural slot |
| **UAG** [HF blog] | any target/assistant, different tokenizers; 1.5–2× (server) | detokenize→retokenize + context window; **sampling-only** | **cross-vocab** | none | ⚠️ **naive cross-tokenizer = 0.58–0.70× (SLOWDOWN) on Apple Silicon** + extra k+1 BW pass [2604.16368] | cross-tokenizer slot — **must be router-gated** |
| **SpecInfer** [2305.09781] | token-tree + parallel verify (the foundational primitive) | tree KV / masks | n/a | n/a | tree verify on Metal is an open build | the verifier architecture itself |

**Not verified in this pass (treat as unconfirmed, not as fact):** the specific *"EAGLE-3.1"*
mechanisms (FC-norm / post-norm feedback / attention-drift fix), the router-policy papers
(LTD, PEARL, TETRIS, BanditSpec), and Medusa/Hydra/MTP heads — no primary source surfaced.
Pursue them as *hypotheses to validate*, not settled technique.

## 2. What Hawking already has + the kill ledger (respect it)

The proposal market is mostly a **refactor of existing parts**, not a greenfield build —
`crates/hawking-core/src/speculate/`:
- `user_ngram.rs::UserNgramDraft` — the training-free n-gram proposer (`propose`/`note_token`/`warm_start`).
- `governor.rs::SpecGovernor` — consecutive-miss gating (the proto-router).
- `eagle5.rs` + `eagle5_forward.rs::Eagle5Head` — trained hidden-state draft head (mock + safetensors load + `propose`/`propose_rollout`/tree).
- `replay_oracle.rs` — offline accept-rate scoring (**the oracle gate**).
- `shared.rs::{DraftStats, DraftToken}` — shared types.

**Kill ledger (`docs/dead_levers.md`) — the trained EAGLE-3-like head is recorded NO-GO** and
must not be blindly resurrected:
- τ=0.877 (gate **τ≥2.5**); device accept **6.5%** vs **52% offline** (~8× forward-parity gap).
- Net-negative on Qwen-3B + code: K=2/4/8 → **0.40×/0.30×/0.21×** (worse with bigger K).
- **The free n-gram draft (τ=1.43) beat the trained head (0.877).**
- *Resurrection check:* do NOT retrain the head expecting a win **without an oracle first
  showing achievable τ≥2.5 on the target workload.**

**Why the new frontier is not a blind resurrection — it attacks the exact death causes:**

| recorded death cause | frontier fix | why it's a *distinct* hypothesis |
|---|---|---|
| serial AR drafter overhead (worse at bigger K) | **P-EAGLE / DFlash** parallel-block drafting | removes the K sequential forwards entirely |
| 8× device-vs-offline accept gap (forward drift) | EAGLE-3 multi-layer fusion / *(unverified)* EAGLE-3.1 norm fixes | directly targets the feature/forward mismatch |
| serial verification (one chain) | **DDTree/SpecInfer** tree verify (ancestor-only mask, single pass) | verifies many futures per target pass |
| **fast** target (Qwen-3B @ 36.9 tps = worst case) | bigger/slower targets | EAGLE-3 *gains* as batch/target cost grows in our regime |
| **no router** — always ran spec, always paid | the `expected_gain` router | **disables spec the instant it stops paying** |

**The single biggest insight:** the old head wasn't only a weak draft — it had **no router**.
The router would have *disabled* spec on fast Qwen-3B automatically. So the router is the thing
that makes a previously-net-negative head **safe to even attempt**, and it is fully consistent
with the ledger: n-gram stays the live base; any trained head stays gated behind τ≥2.5.

## 3. Unified interface

```rust
trait Proposer {
    fn name(&self) -> &str;
    fn cost_estimate(&self, ctx: &Ctx, budget: Budget) -> CostNs;   // predicted draft ms
    fn requires_hidden(&self) -> bool;        // EAGLE-family: taps target hidden states
    fn requires_text_bridge(&self) -> bool;   // cross-tokenizer: detok→retok needed
    fn propose(&mut self, ctx: &Ctx, budget: Budget, tel: &Telemetry) -> Proposal;
}

enum Proposal {
    TokenLine(Vec<u32>),                 // n-gram, AR draft
    TokenTree { nodes, ancestor_mask, position_ids },  // SpecInfer/DDTree
    TextSpan(String),                    // pre-retokenization
    CrossTokenizerSpan { text, src_vocab },
    RetrievalSpan { tokens, source },
}

struct ProposalRouter { /* per-proposer telemetry, hysteresis state, policy */ }
struct Verifier { /* exact target path; accepts longest valid prefix / tree path */ }
struct Telemetry {
    accepted, rejected, draft_ns, verify_ns, retokenize_ns, sync_ns,
    target_ms_per_token, context_class, entropy, prompt_class, recent_accept_hist,
}
```

`Eagle5Head` already exposes `propose`/`propose_rollout`/tree; `UserNgramDraft` already exposes
`propose`/`note_token` — both become `impl Proposer`. `SpecGovernor` folds into the router's
hysteresis state. `replay_oracle` becomes the **offline promotion gate** every neural proposer
must clear (τ≥2.5) before the router is even allowed to enable it at runtime.

## 4. Router decision rule (the novelty)

Enable a proposer for the next step only when its **expected wall-clock gain clears a margin**:

```
expected_gain =  E[accepted_tokens] · target_ms_per_token
              −  draft_ms  −  verify_extra_ms  −  retokenize_ms  −  synchronization_ms
```

- `E[accepted_tokens]` from EAGLE-2-style **context-dependent** confidence (acceptance is
  context-dependent, not just position-dependent — [2406.16858]); drives dynamic draft
  length / tree width.
- The **Metal terms matter most** (`draft_ms`, `synchronization_ms`): the research's open
  question is precisely whether per-step Metal launch/sync erodes the gains parallel/tree
  drafting buys. So the router optimizes **tokens-accepted-per-wall-clock-second**, *measured
  on Metal*, never raw acceptance.
- **Hysteresis:** disable a proposer fast on a rejection streak; re-enable only after a cool-down
  + a probe (prevents thrash). This generalizes the existing `SpecGovernor`.
- **Cross-tokenizer is gated hard:** start *disabled*; the router may enable a `TextSpan`/
  `CrossTokenizerSpan` proposer only if measured `expected_gain > 0` on Metal — the
  0.58–0.70× slowdown evidence [2604.16368] means naive translation is net-negative by default.

## 5. Kill / resurrection checklist (per proposer)

| proposer | promote when | kill when | resurrection gate |
|---|---|---|---|
| n-gram / suffix | always-on base (τ=1.43 already best) | never (free, lossless) | n/a |
| REST/session datastore | `expected_gain>0` on repo/agent loops | datastore stale / lookup ms > savings | rebuild cache |
| EAGLE-3-H (trained) | **offline τ≥2.5 on the target workload FIRST** | net-negative on Metal paired-bench | a *new* mechanism (parallel/tree/norm-fix), not the dead AR-serial path |
| P-EAGLE-H / DFlash-H | parallel head clears τ≥2.5 **and** measured Metal gain | AR-overhead removal doesn't beat n-gram | per-target |
| tree verify | single-pass verify beats serial on Metal | mask/KV reshape cost > tree benefit | DDTree fixed-node-budget design |
| cross-tokenizer (UAG/OmniDraft) | measured `expected_gain>0` on Metal | default (0.58–0.70× slowdown) | only if bridge overhead < target-cost savings on a *slow* target |

## 6. Phased build (model-free first; neural only after oracle gates)

- **Phase 0 — unify.** One `Proposer`/`Router`/`Verifier`/`Telemetry` layer; fold `SpecGovernor`
  + the two serve/core governors into shared telemetry. *(Refactor of existing code.)*
- **Phase 1 — universal base.** n-gram default-on; add suffix-array/SAM; warm-start from
  prompt/session/repo tokens. (Builds on `UserNgramDraft`.)
- **Phase 2 — retrieval.** REST-style local datastore over repo + chat history + generated text.
- **Phase 3 — router v1.** Cost-aware arbitration between n-gram, REST, and no-spec via
  `expected_gain` + hysteresis. **First shippable differentiator.**
- **Phase 4 — neural, gated.** EAGLE-3-H prototype: hidden capture at swept layer triplets,
  *(test the unverified EAGLE-3.1 norm fixes as a hypothesis for the forward-parity gap)*,
  pruned-vocab head — **strict offline τ≥2.5 oracle before any runtime promotion.**
- **Phase 5 — parallel.** P-EAGLE-H or DFlash-H: kill AR drafter overhead (the recorded
  bigger-K death cause) on Metal.
- **Phase 6 — tree verify.** Position IDs + ancestor-only masks + KV for DDTree-style single-pass
  tree verification on Metal (the key open Metal build).
- **Phase 7 — cross-tokenizer bridge.** UAG/OmniDraft text-span translation + online n-gram
  mapping — **router-gated**, never default-on, given the slowdown risk.
- **Phase 8 — online policy.** Contextual-bandit / RL-lite router trained from Hawking's own
  accept/reject telemetry. *(Policy papers unverified — research before adoption.)*

## 7. Benchmark matrix (every promotion is Metal-measured, paired)

Targets: Qwen 3B (control / fast = worst case), Qwen 7B/14B, Llama 8B/70B (if fits quantized),
an MoE target. Workloads: **code, JSON/tool-calls, agent loops** (n-gram-favorable), long
context, prose, reasoning (neural-favorable). Metrics per cell: accept rate, draft ms, verify
ms, retokenize ms, sync ms, **effective tokens/s vs baseline**, quality (exact-verify ⇒ lossless
by construction; confirm bit-identity). The whole point: find *where each proposer's
`expected_gain>0`* on **this** hardware — the map the published CUDA numbers cannot give us.

## 8. Open problems = the research bets

1. **Metal wall-clock reality of EAGLE-3 / P-EAGLE / DFlash** — all published numbers are
   CUDA/server; per-step Metal launch+sync may erode them. (An MLX DFlash port exists,
   unverified.) *This is the measurement only Hawking is positioned to do.*
2. **Cheap Metal tree verify** — position IDs, ancestor-only masks, KV layout for *dynamic*
   (context-dependent) trees without static-kernel reshape cost under unified-memory pressure.
3. **The online router policy** — contextual-bandit/hysteresis to maximize tokens-accepted-
   per-wall-clock-second; the named policy papers were unverified and need a primary-source pass.
4. **Can cross-tokenizer neural drafting *ever* be net-positive on Metal?** (vs 0.58–0.70×) —
   or should heterogeneous targets be served only by the tokenizer-native n-gram base locally?

---

### Sources (claim-verified)
[EAGLE 2401.15077] · [EAGLE-2 2406.16858] · [EAGLE-3 2503.01840] · [SpecInfer 2305.09781] ·
[P-EAGLE 2602.01469] · [DFlash 2602.06036] · [DDTree 2604.12989] · [OmniDraft 2507.02659] ·
[REST 2311.08252] · [Medusa 2401.10774] · [prompt-lookup github/apoorvumang] ·
[UAG huggingface.co/blog/universal_assisted_generation] · cross-family-on-Apple-Silicon 2604.16368.
*DFlash 6×/2.5× headline REFUTED (0-3); EAGLE-3.1 + router-policy papers (LTD/PEARL/TETRIS/
BanditSpec) + Medusa/Hydra/MTP unverified in this pass — pursue as hypotheses.*
