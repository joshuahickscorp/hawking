# 06 · Model & Inference Orchestration

> **Purpose (one line).** Because Hawking owns the runtime, the weights, *and* the quantizer, the decoder is not a sealed cloud endpoint we prompt and pray to — it is an **instrument we control token-by-token**: a fleet of co-designed **model roles** behind a confidence-aware **router**, a **grammar/constrained-decode service** that makes small models emit valid tool-calls *by construction*, **custom samplers** dialed per task, **logit-level signals** (confidence, entropy, self-certainty, best-of-N) handed to the agent, **speculative decoding** for free latency, **LoRA hot-swap** per language/task, and a **fine-tune-at-condense flywheel** that personalizes the local model on the user's own accepted edits — none of which a metered cloud API will ever let you touch.

---

## Table of contents

1. [Purpose & scope](#1-purpose--scope)
2. [Tenets](#2-tenets)
3. [State of the art + limits (cited)](#3-state-of-the-art--limits-cited)
   - 3.1 [Speculative decoding](#31-speculative-decoding)
   - 3.2 [Constrained / structured generation](#32-constrained--structured-generation)
   - 3.3 [Model routing & difficulty estimation](#33-model-routing--difficulty-estimation)
   - 3.4 [Samplers](#34-samplers)
   - 3.5 [Logit-level confidence & test-time compute](#35-logit-level-confidence--test-time-compute)
   - 3.6 [LoRA serving & hot-swap](#36-lora-serving--hot-swap)
   - 3.7 [Activation steering / representation engineering](#37-activation-steering--representation-engineering)
4. [The Hawking design (concrete)](#4-the-hawking-design-concrete)
   - 4.1 [Architecture & where this layer sits](#41-architecture--where-this-layer-sits)
   - 4.2 [The model-role system](#42-the-model-role-system)
   - 4.3 [The role registry (schema)](#43-the-role-registry-schema)
   - 4.4 [The routing policy (decision algorithm)](#44-the-routing-policy-decision-algorithm)
   - 4.5 [Constrained / grammar decode as a first-class service](#45-constrained--grammar-decode-as-a-first-class-service)
   - 4.6 [Sampler profiles](#46-sampler-profiles)
   - 4.7 [Logit-level features exposed to the agent](#47-logit-level-features-exposed-to-the-agent)
   - 4.8 [Speculative decode strategy](#48-speculative-decode-strategy)
   - 4.9 [LoRA / adapter hot-swap](#49-lora--adapter-hot-swap)
   - 4.10 [The fine-tune-at-condense personalization flywheel](#410-the-fine-tune-at-condense-personalization-flywheel)
   - 4.11 [Multi-model concurrency & energy/thermal/RAM-aware scheduling](#411-multi-model-concurrency--scheduling)
5. [How we EXCEED — "cloud literally cannot do this"](#5-how-we-exceed)
6. [Failure modes & mitigations](#6-failure-modes--mitigations)
7. [Extensibility / plugin points](#7-extensibility--plugin-points)
8. [Bleeding-edge / moonshots (ranked)](#8-bleeding-edge--moonshots-ranked)
9. [Open questions & dials](#9-open-questions--dials)
10. [Cross-references](#10-cross-references)
- [Appendix A — Binding contracts (schemas other chapters import)](#appendix-a--binding-contracts)
- [Appendix B — Prioritized runtime-hooks ask-list](#appendix-b--prioritized-runtime-hooks-ask-list)
- [Appendix C — Source register](#appendix-c--source-register)

---

## 1. Purpose & scope

This chapter specifies the **model and inference layer** of HIDE — the subsystem that turns "a running language model on this Mac" into "a controllable, multi-role, decode-level-programmable inference service the agent and tools drive." It is the chapter where the project's structural thesis pays off: *Hawking owns the whole stack — `hawking-core` (the engine), `hawking-serve` (the HTTP/batching surface), the model weights, and `Hawking Condense` (the `.tq` sub-4-bit quantizer).* That ownership is the entire moat. A cloud coding agent rents tokens from a sealed `/chat/completions` box: it cannot read a logit, cannot install a grammar, cannot swap a sampler, cannot hot-load a LoRA, cannot fine-tune the model on your diffs, and cannot run six models at once for free. **We can do all of it**, and this chapter is the design for doing it deliberately.

The animating idea, stated plainly: **small local models become reliable not by being smarter but by being *constrained and instrumented*.** A 7B model that is *forced* to emit schema-valid tool-calls, whose token confidence the agent can *read*, whose sampler is *deterministic for edits*, and which has a *language-specific LoRA* loaded, punches far above a 7B model you merely prompt politely over HTTP. The decoder is where that leverage lives, and we own the decoder.

### In scope

- **The model-role system** — hero/coder, fast-draft, embedder, reranker, summarizer/compactor, classifier/router: each a *served role* with a descriptor, a default sampler profile, and a footprint.
- **The routing policy** — task→role mapping, difficulty estimation, escalation on low confidence/high entropy, the decision algorithm (pseudocode).
- **Constrained / grammar decode as a service** — compiling a JSON-schema / GBNF grammar / edit-format / plan-schema down to the existing `json_constrain.rs` `mask_logits` machinery; making tool-calls valid by construction.
- **Custom sampler profiles** — per-task sampler specs (deterministic for edits, exploratory for brainstorming), extending the in-tree `SamplingParams` with min-p / typical / DRY / logit-bias.
- **Logit-level features** — token confidence, entropy gating, self-certainty, self-consistency voting, logprob-guided best-of-N, exposed as agent-callable signals (ch.02).
- **Speculative decode strategy** — the in-tree proposer fleet (ExactShared, Eagle5, n-gram, suffix, retrieval) + the wall-clock router; when spec wins for agent loops; self-speculation.
- **LoRA / adapter hot-swap** — per-language/per-task adapters, S-LoRA-style serving, the selection seam.
- **The fine-tune-at-condense personalization flywheel** — teaching the local model HIDE's tool protocol and the user's accepted-edit style.
- **Multi-model concurrency + energy/thermal/RAM-aware scheduling** on Apple Silicon.
- **The prioritized runtime-hooks ask-list** to the runtime team (Appendix B).

### Out of scope / deferred (with explicit gates)

| Item | Where it lives | Status |
|---|---|---|
| `.tq` sub-4-bit serving on GPU, 32B residency, the GPU bitslice GEMV kernel | *Hawking Condense* + `hawking-core/tq_gpu.rs` | **Runtime testing / LATER, NOT shell-gating.** This chapter designs *how HIDE exploits* a `.tq` model once served and specifies the role/footprint hooks, but does not finish `.tq` serving. The CPU `.tq` path is the parity oracle; the GPU kernel is staged. |
| Hawking-HF model distribution (publishing condensed models) | *Hawking Condense* / packaging | **Deferred until 32B ready** (per ch.01). The role registry references model artifacts by id; it does not depend on a published registry. |
| The subprocess **model router process** (running N models, one-per-process) | designed in ch.01 (`ModelProvider`/`ProviderCaps`) + here | This chapter **specifies the routing *policy* and the *roles* the router exposes**; the process-management mechanism (spawn/supervise/route) is ch.01's. |
| Training infrastructure for adapters / personalized checkpoints (the actual fine-tune job runner) | *Hawking Condense* tooling (`tools/training/`) | **Roadmap.** §4.10 designs the *data flywheel and contracts*; the trainer that consumes the dataset is Condense's. |

> **Scoping rule, restated as a hard invariant.** Everything in §4 that requires *new runtime internals* (logprob streaming, a `grammar` request field, LoRA selection, KV handles, a request-time sampler superset) carries a **[RUNTIME-SIDE — LATER]** tag and a **[SHELL-TODAY]** fallback that works against the **current localhost OpenAI-compatible surface** — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/hawking/generate`, `/v1/hawking/tokens`, `/metrics` — plus the in-tree `json_mode` flag. **v1 of HIDE binds a stable localhost HTTP surface; the deeper decoder hooks are a prioritized roadmap (Appendix B), never a blocker.**

### Ground truth this chapter binds to (verified in-tree)

This chapter binds to **real types**, not aspirations. Verified by reading the source:

- **Engine seam** (`crates/hawking-core/src/engine.rs`): the `Engine` trait exposes `generate`, `embed` (powers `/v1/embeddings`), `model_id`, `model_arch`, `encode_prompt_for_batch`/`decode_token_for_batch`, the continuous-batch seams (`prefill_slot`, `prefill_slot_from_pos`, `copy_kv_prefix_to_slot`, `forward_multiseq_*`, `kv_fingerprint_at_pos`), and `GenStats::dec_tps()` / `draft_accept_rate()`. `GenerateRequest` already carries `json_mode: bool` and `abort`/`max_stall_ms`. **`SamplingParams { temperature, top_k, top_p, repetition_penalty, seed }`** is the current knob set — *narrower* than what this chapter needs, so §4.6 specifies the superset.
- **Sampler** (`sample.rs`): CPU reference does rep-penalty → temperature → softmax → top-K → top-P → draw; greedy at `temperature ≤ 0`. The Metal path keeps logits on-GPU and **only the sampled token id crosses the bus** — the key fact that makes logprob/confidence readback a *deliberate, gated* feature (§4.7), not free.
- **Constrained decode** (`json_constrain.rs`): a `JsonConstraint` state machine + `JsonVocabIndex` (token_id→text, built once per model) + `mask_logits(&vocab, &mut logits)` that sets invalid-continuation tokens to `-inf` before sampling, advanced by `advance(&text)`. **This is the exact machine §4.5 compiles arbitrary grammars onto.** Today it hard-codes generic JSON; the design generalizes the *valid-next-token mask* to schema/grammar-driven.
- **Speculative decode** (`speculate/`): a full fleet — `SpeculateMode { Off, ExactShared, Eagle5 }` on `EngineConfig`; proposers `UserNgram`, `SuffixArray`, `Eagle5`, `Rest`, `CrossTokenizer`, `Retrieval`, `Tree`, `ParallelDraft`; a **wall-clock `router.rs`** with a measured `verify_cost_forwards(b)` curve (B=8 ≈ 4.15 forward-units, the ~1.93× ideal point) and per-proposer EWMA cost models; a `SpecGovernor` (rolling accept-rate auto-disable, default min 0.35, 5 consecutive-reject ceiling) and a **UCB1 contextual-bandit `policy.rs`**. Verify is the full model → **greedy output is bit-identical to no-spec at temperature 0** (the losslessness guarantee). *Caveat carried forward from memory: batched verify ≠ greedy kernel at near-ties — the EH property gate caught this; the greedy-lossless guarantee holds for the ExactShared/Eagle5 verify path; any tree/parallel-draft mode that risks it stays gated.*
- **Serve surface** (`hawking-serve/src/http.rs`): the six routes above; `ResponseFormat {"type":"json_object"}` already wires `json_mode` through `chat_completions`. **No `logprobs`, no `grammar`, no `tools`-strict, no LoRA field yet** — Appendix B asks for them.
- **`.tq` / Condense binding** (`tq.rs`): a `.tq` file is the absorbed strand-quant `STR2` format; CPU decode is the bit-exact parity oracle the staged GPU bitslice GEMV must reproduce. *Hawking Condense* is the sibling product that *produces* `.tq` and the personalization checkpoints (§4.10).
- **Models present**: Qwen-dense (primary hero/coder), RWKV-7 & Mamba-2 (SSM flat-decode long-context), Llama, DeepSeek-V2, Gemma2, Phi3, Mixtral/Qwen-MoE/OLMoE. **Multiple architectures already load** — the substrate for a multi-role fleet.

Where this chapter says "we already have X," it was checked against these files. Where it says "[RUNTIME-SIDE — LATER]," the seam exists but the body is `todo!()`/env-gated/absent, and a **[SHELL-TODAY]** fallback is given.

---

## 2. Tenets

1. **Own the decoder, or you own nothing.** Every advantage in this chapter — grammar-valid tool-calls, logit confidence, custom samplers, LoRA hot-swap, personalization — exists *only* because we control the token sampling loop. A cloud API hands you text; we hand the agent the *distribution*. This is the moat; everything else is engineering around it.

2. **Constrain, don't hope.** The reliable way to get a small model to emit a valid tool-call / plan / JSON / edit is to make every invalid token *unreachable* at decode time. Validation-and-retry is the cloud's only tool; **construction** is ours. Structured generation is a *correctness mechanism*, not a convenience. ([Dong et al. 2024], [Beurer-Kellner et al. — LMQL/Outlines], [JSONSchemaBench 2025])

3. **Right model, smallest sufficient.** A task gets the *cheapest role that clears its bar*, with **escalation on measured uncertainty**, not a fixed "always use the big one." We have the unfair ability to read the model's own confidence to make that call locally and for free (RouteLLM does it with a trained external router; we can do it with logits). ([Ong et al. 2024])

4. **Determinism where it matters; exploration where it helps.** An edit, a refactor, a tool-call must be *reproducible* (greedy/seeded, constrained). Brainstorming, naming, and search benefit from *diversity*. The sampler is a **per-task profile**, not a global default. We never apply creative-writing temperature to a code patch.

5. **Speculation is free latency when it clears the curve, and silent when it doesn't.** We already measure the wall-clock cost of verify vs. the value of an accepted token and auto-disable when spec stops paying (the router + governor). Spec is *always lossless* on the verified greedy path. It is a throughput dial, never a quality risk.

6. **Logits are an API.** Token confidence, entropy, self-certainty, n-best logprobs, and per-position margins are *first-class signals* the agent and the constrained-decode service consume — to gate, to vote, to escalate, to detect hallucination, to know when to stop. Cloud logprobs are lossy or absent; ours are exact and free to compute (the cost is *moving* them off-GPU — a gated readback, §4.7).

7. **The model is co-designed with the harness.** Because *Hawking Condense* makes the weights, we can **teach the model HIDE's exact tool protocol and the user's edit style at quantization/fine-tune time**. The harness and the model evolve together. Cloud models are frozen strangers; ours is a colleague we train. This is the flywheel (§4.10).

8. **One process per model today; a fleet behind a router.** The runtime serves one model per process (ground truth). Multi-role = a **subprocess router** (ch.01) over several `hawking-serve` instances. We design the *roles and the routing policy*; we exploit the local fact that **idle silicon is free** — running a 0.5B draft + a 7B hero + an embedder concurrently costs only power, not dollars.

9. **Lossless by default, lossy by opt-in and gated.** Spec-decode greedy is bit-identical; constrained decode never changes a *valid* token's relative ranking, only masks invalid ones; aggressive samplers and lossy roles are per-task and oracle-gated. The correctness-critical path always has a deterministic, unconstrained-except-by-grammar escape.

10. **Energy and thermals are a scheduling input, not an afterthought.** On a laptop, the model layer is a *power budget*. Role selection, concurrency, and spec aggressiveness are throttled by thermal headroom and battery state. A cloud has no equivalent concern; we turn it into a feature ("quiet mode," "on-battery mode").

11. **Hooks over forks.** Where the runtime can't do something yet, we specify the *minimal hook* (a request field, a readback flag, an FFI seam) and ship a **[SHELL-TODAY]** approximation against the current HTTP surface. The shell never blocks on a kernel landing; the ask-list (Appendix B) is prioritized by leverage.

---

## 3. State of the art + limits (cited)

Tagged **[PROVEN-IN-PROD]** / **[RESEARCH-PROVEN]** / **[SPECULATIVE]**, with *difficulty* (build cost for us) and *impact*. Full source register in [Appendix C](#appendix-c--source-register).

### 3.1 Speculative decoding

| Technique | Mechanism (compressed) | Headline | Maturity | Diff / Impact |
|---|---|---|---|---|
| **Classic spec-decode (draft+verify)** | A small draft model proposes K tokens; the target verifies in one batched forward; accept the longest matching prefix (rejection sampling preserves the target's distribution). | 2–3× decode speedup, **output-distribution-preserving**. | [PROVEN-IN-PROD] | med / high. ([Leviathan et al. 2023], [Chen et al. 2023b]) |
| **Medusa** | Add K extra decoding *heads* to the target itself (no separate draft model); tree-attention verifies multiple candidate continuations at once. | ~2.2–2.8× with trained heads; no second model to host. | [PROVEN-IN-PROD] | med (trains heads) / high. ([Cai et al. 2024b]) |
| **EAGLE-2 / EAGLE-3** | Draft at the *feature* (hidden-state) level, not token level; **EAGLE-3** switches to **direct token prediction + "training-time test"** (draft trained on its own rollouts) and finds a *scaling law* (more draft-training data → proportionally more speedup). | EAGLE-3: **3–6.5×**; acceptance length ~4.5–5.0 tokens/cycle, near-**flat across positions** (EAGLE drops). Trained on ~532K examples (ShareGPT+UltraChat). | [PROVEN-IN-PROD] | high (trains a head) / very high. ([Li et al. 2024b], [Li et al. 2025 — EAGLE-3]) |
| **Self-speculative / layer-skip** | The model drafts *itself* by skipping layers / early-exit, then verifies with full depth. No extra weights, no extra memory. | ~1.3–1.8×, zero extra parameters. | [RESEARCH-PROVEN] | med / med. ([Elhoushi et al. 2024 — LayerSkip], [Zhang et al. 2023 — self-spec]) |
| **Lookahead decoding** | Jacobi-style parallel n-gram generation + verification; no draft model. | ~1.5–2× on code, draft-free. | [RESEARCH-PROVEN] | med / med. ([Fu et al. 2024]) |
| **REST / retrieval drafting** | Draft tokens come from a *datastore retrieval* (suffix-automaton / corpus) rather than a model. | Strong on repetitive/code text; near-zero draft cost. | [RESEARCH-PROVEN] | low–med / med. ([He et al. 2024]) |
| **Scaling laws for spec-decode** | Speedup grows with draft quality/training in a predictable curve; governs the cost/benefit of investing in a better draft. | Formalizes the "is a bigger draft worth it" tradeoff. | [RESEARCH-PROVEN] | — / planning. ([arXiv:2505.07858]) |

> **In-tree reality.** Hawking already ships the *fleet and the arbiter*: 8 proposers, a measured `verify_cost_forwards` curve, EWMA per-proposer cost models, a rolling-accept governor, and a UCB1 bandit policy — i.e. the **wall-clock-optimizing router is more sophisticated than most published single-proposer systems**. The gaps are: a *trained* Eagle5/EAGLE-3 head (mock head exists; real accept rate needs training), and the cross-tokenizer path for using a *different-family* draft (e.g. a 0.5B drafting for a 7B). The memory caveat stands: **batched verify can diverge from the greedy kernel at near-ties** → lossless guarantee is enforced only on the ExactShared/Eagle5 greedy-verify path; risky tree/parallel modes stay gated behind the EH property test.

### 3.2 Constrained / structured generation

| Technique | Mechanism | Headline | Maturity | Diff / Impact |
|---|---|---|---|---|
| **GBNF grammars (llama.cpp)** | A BNF-like grammar; at each step a stack machine computes the set of valid next tokens; mask the rest to `-inf`. | The de-facto local-grammar format; forces any CFG-describable output. | [PROVEN-IN-PROD] | low (we have the masking primitive) / high. (llama.cpp grammars) |
| **Outlines (regex/JSON→FSM)** | Compile a regex or JSON-schema to a **finite-state machine**; precompute, per FSM state, the allowed-token set; O(1) mask lookup at decode. | Near-zero per-token overhead after compile; guarantees schema validity. | [PROVEN-IN-PROD] | med / high. ([Willard & Louf 2023]) |
| **XGrammar** | Split vocab into **context-independent** tokens (pre-checked at compile) vs **context-dependent** (checked at runtime via a **pushdown automaton**); compiler-style inlining + state merging; parallel grammar compile; persistent execution stack. | **Up to 100× faster** grammar execution; **near-zero overhead** structured generation in end-to-end serving; integrated into vLLM/SGLang/TRT-LLM/MLC. **XGrammar-2 (2026): up to 80× over XGrammar, cross-grammar caching, JIT.** | [PROVEN-IN-PROD] | med–high / very high. ([Dong et al. 2024], [XGrammar-2 2026]) |
| **JSON-schema decoding (productized)** | OpenAI "Structured Outputs" / strict tool-calling; Anthropic/Google/Bedrock equivalents 2024–2026. **Strict mode guarantees args match the schema**, removing a whole class of tool-use failures. | Commoditized "schema-valid by construction"; the *baseline expectation* now. | [PROVEN-IN-PROD] | — / high (sets the bar we must meet/beat locally). |
| **Pre³ / DPDA** | Use a **deterministic** pushdown automaton for the grammar to cut per-step mask cost further. | Faster structured generation via determinism. | [RESEARCH-PROVEN] | high / med. ([arXiv:2506.03887]) |
| **Coverage/quality studies** | JSONSchemaBench, "Generating Structured Outputs… Benchmark" — measure *empirical* schema-validity and the (sometimes negative) effect of constraints on *task* quality. | Constraints can **hurt reasoning** if they fight the model's natural token order ("thinking before constraining" matters). | [RESEARCH-PROVEN] | — / planning. ([JSONSchemaBench 2025], [arXiv:2601.07525]) |

> **In-tree reality.** `json_constrain.rs` already does the load-bearing primitive — **per-state valid-next-token masking before sampling**. What it lacks is (a) a *schema/grammar* driving the state machine instead of hard-coded generic JSON, and (b) the XGrammar-style **compile-once, split context-independent/dependent** optimization so the mask is cheap. §4.5 specifies exactly that generalization, reusing `mask_logits`. **The limit to respect:** over-constraining can degrade reasoning — so HIDE separates a "think" (unconstrained) phase from a "emit" (constrained) phase, and never grammars the chain-of-thought.

### 3.3 Model routing & difficulty estimation

- **RouteLLM** ([Ong et al. 2024], ICLR 2025). Trained routers (matrix-factorization, BERT, causal-LLM classifier) decide strong-vs-weak model per query from preference data. **Matrix-factorization router: 95% of GPT-4 quality at 26% GPT-4 calls (~48% cheaper); with LLM-judge-augmented data, 95% quality at 14% calls (~75% cheaper).** The lesson: *most queries don't need the big model*, and a cheap router captures most of the savings. The limit: their router is an *external* model trained offline; it has no access to the strong model's own uncertainty.
- **Task-based / capability routing** (2024–2026 surveys, LLMRouterBench). Route by detected task type (codegen vs chat vs extraction), by predicted difficulty, or by a cascade (try cheap, escalate on a verifier signal). Cascades dominate when a cheap, reliable *escalation signal* exists.
- **Difficulty estimation.** Proxies: prompt length/complexity, presence of multi-step reasoning markers, retrieval-coverage, *and* — uniquely available to us — **the model's own first-token entropy / margin** (a high-entropy first token predicts a hard generation). ([self-certainty work, §3.5])
- **HIDE's unfair angle.** We don't need an external trained router to *start*: we run the cheap role and **read its confidence**; low confidence / high entropy / failed self-consistency *is* the escalation signal (a cascade with a free, exact gate). A learned router (RouteLLM-style) is an optimization we add later, trained on our own accept/edit telemetry.

### 3.4 Samplers

| Sampler | Mechanism | When it wins | Maturity | Note |
|---|---|---|---|---|
| **Greedy / low-temp + seed** | argmax or near-argmax, fixed RNG. | Edits, refactors, tool-calls, anything that must be **reproducible**. | [PROVEN-IN-PROD] | The in-tree default path at `temp≤0`. Our edit profile. |
| **Top-k / top-p (nucleus)** | Truncate to k or to cumulative-p mass. | General chat; the current `SamplingParams`. | [PROVEN-IN-PROD] | Already shipped. |
| **min-p** | Dynamic threshold = `p_min × max_prob`; keeps the pool tight when the model is confident, wide when it isn't. | **High-temperature creative/coherent** balance; better than top-p at temp>1. Gains shown 1B–123B on GPQA/GSM8K/creative. | [RESEARCH-PROVEN] | **Contested**: a 2026 human-eval critique finds min-p does *not* beat baselines on quality/diversity Pareto. → ship as an *option*, default off, A/B locally. ([Nguyen et al. 2024], [arXiv:2506.13681]) |
| **Typical / locally-typical** | Keep tokens whose surprisal is near the distribution's entropy. | Reduces degenerate repetition; natural-text feel. | [RESEARCH-PROVEN] | med build. ([Meister et al. 2023]) |
| **DRY (Don't Repeat Yourself)** | Penalize tokens that would *extend a repeated n-gram* (sequence-aware, unlike flat rep-penalty). | Kills loops/boilerplate repetition without nuking legit repeats (e.g. code). | [RESEARCH-PROVEN] (community-proven) | low–med build; **strictly better than flat rep-penalty for code**. |
| **mirostat** | Feedback loop targeting a fixed *perplexity* (surprise) setpoint. | Long-form with controlled "creativity"; avoids boredom/incoherence drift. | [RESEARCH-PROVEN] | med; entropy-based, more tuning. ([Basu et al. 2021]) |
| **logit-bias / banned tokens** | Add/subtract per-token bias; ban tokens outright. | Forbid `eval(`, force/forbid specific identifiers, steer formatting. | [PROVEN-IN-PROD] | trivial atop `mask_logits`; a free safety/format lever. |

> **In-tree reality.** `SamplingParams` covers temp/top-k/top-p/rep-pen/seed and a clean CPU+Metal sampler. The superset HIDE wants (min-p, typical, DRY, logit-bias, per-request) is a *small additive struct* (§4.6) — the sampler loop is already the right shape (mask → penalize → truncate → draw). **The limit:** sampler choice is mostly a *quality-of-life / diversity* lever; for correctness we lean on *constraint + determinism*, not fancy sampling.

### 3.5 Logit-level confidence & test-time compute

- **Self-consistency** ([Wang et al. 2023b]). Sample N reasoning paths, majority-vote the answer → large accuracy gains. The canonical test-time-compute lever.
- **Self-certainty / logprob confidence** ([arXiv:2502.18581], [Deep Think with Confidence 2025]). An LLM's *own* probability distribution encodes certainty; **self-certainty (a divergence-from-uniform metric) separates correct from incorrect better than perplexity**, and enables *Best-of-N by confidence* and *early-stopping*. Prefix-confidence at test time improves math reasoning efficiently ([arXiv:2507.18122]).
- **Self-calibration** ([arXiv:2503.00031]). Estimate calibrated confidence in *one forward pass* → efficient early-stop for Best-of-N / self-consistency.
- **Verifier-guided Best-of-N** ([arXiv:2502.20379], [arXiv:2505.04842]). A separate verifier (or an ensemble of heterogeneous verifiers) scores N candidates; scaling the *number/type of verifiers* is itself a test-time axis.
- **HIDE's unfair angle.** All of these need **logprobs / the distribution**, which cloud APIs expose lossily (top-k logprobs, often disabled) or not at all. **We compute them exactly, on-device, for free** — the only cost is *moving* a logit row off the GPU (the sampler keeps logits on-GPU and reads back 4 bytes; full readback is `vocab×4` bytes/step — a *deliberate, gated* path, §4.7). So entropy gating, self-certainty escalation, and confidence-weighted best-of-N are native HIDE capabilities, not API features we beg for.

### 3.6 LoRA serving & hot-swap

- **S-LoRA** ([Sheng et al. 2024], MLSys). Store *all* adapters in host RAM, fetch the active ones to GPU on demand; **Unified Paging** (one memory pool for ranked adapter weights + KV of varying lengths) + heterogeneous batching + custom kernels. **Serves thousands of adapters on one GPU; up to 4× throughput vs naive PEFT/vLLM and orders-of-magnitude more adapters.** The blueprint for "many small task-specific adapters, swapped cheaply."
- **Punica / multi-LoRA batching** (2023–2024). Batch requests for *different* LoRAs in one kernel (SGMV) so a fleet of adapters shares a base-model forward.
- **The local fit.** On a single-user box we don't need *thousands*; we need a *handful* — per-language (`rust.lora`, `ts.lora`), per-task (`commit-msg.lora`, `sql.lora`), and the *personal* adapter (§4.10). Hot-swap is a memory-residency + base-merge decision; **the base model forward is shared, the adapter is a small delta.**
- **In-tree reality.** **No LoRA serving exists yet** (the `lora` grep hits are unrelated test-file substrings). This is a clean **[RUNTIME-SIDE — LATER]** build. §4.9 specifies the selection seam and the **[SHELL-TODAY]** fallback (a *separate served instance* per heavily-used adapter, routed as a role — works now, costs more RAM).

### 3.7 Activation steering / representation engineering

- **Control vectors / CAA** ([Turner et al. 2023 — ActAdd], [Rimsky et al. 2024 — CAA], [Zou et al. 2023 — RepE]). High-level behaviors (refusal, honesty, sentiment, verbosity, format-adherence) are often **linear directions** in the residual stream; add a *steering vector* (difference-of-means over contrastive prompts) at chosen layers to shift behavior — **cheap, train-free, test-time**.
- **The fit + the limit.** Tempting for "make the model more concise," "less likely to hallucinate APIs," "prefer this code style" *without* a fine-tune. But 2026 work ([arXiv:2602.17881]) shows **steering vectors are *unreliable* — geometry-dependent, can break under composition**. → **[SPECULATIVE], moonshot-only (§8)**: an optional, opt-in, oracle-gated steering hook (`steer: [{vector_id, layer, scale}]`), never on the correctness path. We own the residual stream (we run the forward pass), so we *can* do this — but we treat it as a research lever, not a v1 feature.

---

## 4. The Hawking design (concrete)

### 4.1 Architecture & where this layer sits

The model layer is a **fleet of served roles** behind a **router**, fronted by a uniform **inference API** the agent (ch.02) and tools (ch.03) call. Data flows: a *task* arrives → the router picks a *role* (and sampler profile, grammar, adapter) → the request hits a `hawking-serve` instance → tokens stream back *with* the decode-level signals the caller asked for.

```
   ch.02 agent loop / ch.03 tools / ch.04 compiler / ch.05 index
                    │  InferenceRequest { task_kind, prompt/messages,
                    │     sampler_profile?, grammar?, adapter?, want_logprobs?, … }
                    ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ (A) ROUTER  task→role · difficulty est. · escalation · concurrency  │
   │     reads role registry; picks {role, sampler, grammar, adapter}    │
   └───────────────┬───────────────────────────────────────────────────┘
                   │ routes to the chosen role's served instance
                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ (B) ROLE FLEET   (one hawking-serve process per loaded model)       │
   │   hero/coder · fast-draft · embedder · reranker · compactor · router│
   │   each: ModelDescriptor + default SamplerProfile + footprint        │
   └───────────────┬───────────────────────────────────────────────────┘
                   │ /v1/* HTTP today  ·  FFI/handles later
                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ (C) DECODE CORE  (hawking-core Engine)                              │
   │   sampler (sample.rs) · constraint mask (json_constrain.rs→grammar) │
   │   spec-decode router (speculate/) · LoRA delta (later) · KV (ch.04) │
   └───────────────┬───────────────────────────────────────────────────┘
                   │ emits tokens + GenStats + (gated) logprobs/confidence
                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ (D) PERSONALIZATION FLYWHEEL  (Hawking Condense)                    │
   │   accepted-edit dataset → fine-tune/adapter → new role artifact     │
   └───────────────────────────────────────────────────────────────────┘
```

**Module layout** (proposed; binds to existing files where they exist):

```
hawking-orch/                          # NEW crate — shell-side fleet router & policy (HTTP-only deps)
  src/
    registry.rs        # RoleRegistry: load/validate role descriptors (Appendix A.1)
    router.rs          # route(task) -> RouteDecision {role, sampler, grammar?, adapter?} (§4.4)
    difficulty.rs      # difficulty estimators (length, markers, retrieval-coverage, first-token entropy)
    escalation.rs      # confidence/entropy/self-consistency escalation ladder (§4.4, §4.7)
    inference.rs       # InferenceClient: uniform call over /v1/* (today) / FFI (later)
    grammar/           # the constrained-decode SERVICE (§4.5)
      compile.rs       #   schema/GBNF/edit-format -> CompiledGrammar (mask program)
      json_schema.rs   #   JSON-Schema -> grammar
      tool_call.rs     #   tool registry (ch.03) -> per-tool arg grammar
      cache.rs         #   compile-once cache (XGrammar-style), keyed by (grammar_hash, vocab_sig)
    samplers.rs        # SamplerProfile registry + presets (§4.6)
    confidence.rs      # entropy / self-certainty / self-consistency aggregation (§4.7)
    adapters.rs        # LoRA/adapter selection policy (§4.9)
    scheduler.rs       # concurrency + energy/thermal/RAM admission (§4.11)

hawking-core/src/                      # EXISTING — runtime-side decode intelligence
  engine.rs            # Engine trait; GenerateRequest gains grammar/sampler/logprob fields (Appendix B)
  sample.rs            # sampler; gains min-p/typical/DRY/logit-bias (additive) (§4.6)
  json_constrain.rs    # mask_logits primitive; gains grammar-driven valid-set (§4.5)
  speculate/           # the proposer fleet + wall-clock router + governor + bandit (§4.8)
  tq.rs / tq_gpu.rs    # .tq serving (Condense) — role footprint source [LATER]
hawking-serve/src/
  http.rs              # gains optional: grammar param, logprobs, tools-strict, adapter, sampler superset
  spec_gov.rs          # spec acceptance governor (shipped)
```

> **Why a separate `hawking-orch` crate.** The *router, registry, grammar-compiler front-end, sampler-profile registry, escalation policy, and scheduler* are **shell concerns** (per scoping: "shell first"). They depend only on the *HTTP surface* + a tokenizer/vocab dump, so they ship and test without any kernel or `.tq` dependency. The *decode-time mechanisms* (the mask program execution, the spec router, LoRA deltas, logit readback) live in `hawking-core` where the model is. The two meet at `inference.rs`/`grammar/` ↔ the new HTTP fields (Appendix B).

---

### 4.2 The model-role system

A **role** is "a served model specialized for a job, with a default sampler profile, a footprint, and capability flags." The fleet is the set of roles HIDE can route to. **Each role is independently swappable** (a better model for the role drops in by changing its descriptor) — the same pattern ch.05 already assumes for its embedder/reranker roles, and ch.01 exposes via `ModelProvider`/`ProviderCaps`.

The canonical roles:

| Role | Job | Typical model (today) | Default sampler | Why a distinct role |
|---|---|---|---|---|
| **hero / coder** | The main reasoning + code-generation + edit model; handles plans, diffs, hard tool-calls. | **Qwen-dense** (primary), condensed `.tq` for the 32B tier later. | **edit** profile (greedy/low-temp, seeded) for patches; **balanced** for chat. | The capability anchor; everything escalates *to* it. |
| **fast-draft** | The speculative-decode *draft* for the hero (and a standalone responder for trivial completions). | A 0.5B–1.5B Qwen (or the hero's own early-exit / Eagle5 head). | greedy (it only proposes). | Latency: a tiny model drafts, the hero verifies (§4.8). Costs only power. |
| **embedder** | Text/code → vector, for retrieval & memory relevance (ch.04/ch.05). | The hero's `embed()` today; a dedicated code-embedding model later (ch.05's first planned add). | n/a (no sampling). | Different objective; can be a *much* smaller specialized model; called constantly. |
| **reranker** | Precision-rank candidate snippets/memories (ch.05 §re-rank). | **Listwise via the hero** (free, local) today; a cross-encoder later. | greedy + JSON/rank grammar. | Sharpens weak-embedding recall; ch.05 binds this. |
| **summarizer / compactor** | Cheap recursive summarization & context compaction (ch.04 §4.8 `draft_compact`). | A small model or the fast-draft. | balanced, low-temp. | Compaction must be *cheap* and *frequent*; the hero is too expensive to spend on it. |
| **classifier / router-model** | Fast task-type / difficulty / safety classification feeding the router (§4.4). | A tiny model or a constrained single-token forward on the fast-draft. | greedy + enum grammar (one of a fixed label set). | Microsecond decisions; a single constrained token is enough. |
| **(SSM) long-context** | Flat-throughput unbounded-history tasks (watch-this-log, long narrative). | **RWKV-7 / Mamba-2** (in-tree, O(1) state). | balanced. | The architecture *is* the role (ch.04 §4.4.4 routes here by task). |

**Role identity is logical, model binding is physical.** Two roles can map to the *same* served process (e.g. reranker == hero today), or a role can be unfilled (no dedicated embedder → fall back to hero's `embed()`). The registry records the binding; the router resolves logical role → physical endpoint. This is exactly ch.01's `ModelProvider` indirection, specialized to roles.

**Roles are co-designed (the flywheel hook).** Because *Hawking Condense* makes the weights, a role can be a model **fine-tuned for that role** — a hero taught HIDE's tool protocol, a compactor distilled for summarization, a classifier trained on our label set. §4.10 closes this loop.

---

### 4.3 The role registry (schema)

The registry is a versioned, user-inspectable file (`.hide/roles.toml` or JSON) plus built-in defaults. It is a **binding contract** — see [Appendix A.1]. Normative shape:

```jsonc
{
  "schema_version": 1,
  "roles": {
    "hero": {
      "role_kind": "hero",                         // hero|fast_draft|embedder|reranker|compactor|classifier|ssm_long|custom
      "model": {                                   // == ch.04 ModelDescriptor (this chapter PROVIDES it)
        "id": "qwen2.5-7b-instruct",
        "arch": "transformer",                     // "transformer" | "ssm"
        "ctx_len_native": 32768,
        "tokenizer_sig": "blake3:…",
        "footprint_mb": 4600,                      // RAM/VRAM working set (from .tq/Condense later)
        "quant": "q4_k_m",                         // or "tq:str2:3.0bpw" once .tq serves
        "embed_capable": true
      },
      "endpoint": "http://127.0.0.1:8081",         // the hawking-serve instance (router process owns lifecycle)
      "default_sampler": "edit",                   // a SamplerProfile name (§4.6)
      "caps": {                                    // == ch.01 ProviderCaps, role-specialized
        "grammar": true,                           // can enforce a grammar mask
        "logprobs": false,                         // exposes per-token logprobs? [LATER for most]
        "speculative": true,                       // can be a spec-verify target
        "draft_for": ["hero"],                     // (fast_draft) which roles it can draft
        "adapters": ["rust", "ts", "personal"],    // loadable LoRA ids [LATER]
        "max_batch": 8
      },
      "cost": { "ms_per_tok_est": 9.0, "energy_j_per_tok_est": 0.17 },  // for the router's cost model
      "escalates_to": null                          // hero is the top; lower roles escalate UP to it
    },
    "fast_draft": {
      "role_kind": "fast_draft",
      "model": { "id": "qwen2.5-0.5b", "arch": "transformer", "ctx_len_native": 32768,
                 "tokenizer_sig": "blake3:…", "footprint_mb": 500, "quant": "q4_k_m" },
      "endpoint": "http://127.0.0.1:8082",
      "default_sampler": "greedy",
      "caps": { "grammar": true, "logprobs": false, "speculative": false, "draft_for": ["hero"], "max_batch": 8 },
      "cost": { "ms_per_tok_est": 1.5, "energy_j_per_tok_est": 0.03 },
      "escalates_to": "hero"
    },
    "embedder": { "role_kind": "embedder", "model": { "id": "qwen2.5-7b-instruct", "arch": "transformer",
                  "footprint_mb": 0, "quant": "shared" }, "endpoint": "http://127.0.0.1:8081",
                  "default_sampler": null, "caps": { "grammar": false }, "escalates_to": null },
    "reranker": { "role_kind": "reranker", "model": { "id": "qwen2.5-7b-instruct", "footprint_mb": 0,
                  "quant": "shared", "arch": "transformer" }, "endpoint": "http://127.0.0.1:8081",
                  "default_sampler": "edit", "caps": { "grammar": true }, "escalates_to": null },
    "compactor": { "role_kind": "compactor", "model": { "id": "qwen2.5-1.5b", "arch": "transformer",
                  "footprint_mb": 1200, "quant": "q4_k_m" }, "endpoint": "http://127.0.0.1:8083",
                  "default_sampler": "balanced", "caps": { "grammar": true }, "escalates_to": "hero" },
    "classifier": { "role_kind": "classifier", "model": { "id": "qwen2.5-0.5b", "footprint_mb": 0,
                  "quant": "shared", "arch": "transformer" }, "endpoint": "http://127.0.0.1:8082",
                  "default_sampler": "greedy", "caps": { "grammar": true }, "escalates_to": "hero" },
    "ssm_long": { "role_kind": "ssm_long", "model": { "id": "rwkv7-…", "arch": "ssm",
                  "ctx_len_native": 1000000, "footprint_mb": 0, "quant": "…" },
                  "endpoint": "http://127.0.0.1:8084", "default_sampler": "balanced",
                  "caps": { "grammar": true }, "escalates_to": null }
  },
  "routing": { "default_role": "hero", "escalation": "confidence", "budget_mode": "balanced" }
}
```

Notes:
- **`model` is the `ModelDescriptor` ch.04 binds** (id, arch, ctx_len, tokenizer_sig, footprint). This chapter is its *provider* — ch.04 §4.1/§A reference `ModelDescriptor.arch` and `.ctx_len`; this is where they come from. `footprint_mb`/`quant` are populated by *Hawking Condense* once `.tq` serves (until then, the GGUF/quant the model loads with).
- **`caps` is `ProviderCaps`** (ch.01 §4.3) extended with decode-level flags (`grammar`, `logprobs`, `speculative`, `draft_for`, `adapters`). A role advertises what the runtime hook *currently* supports; the router never asks for a cap a role lacks (it degrades — e.g. no `grammar` → validate-and-retry; no `logprobs` → heuristic confidence).
- **`escalates_to`** forms the cascade graph: lower roles point *up* at the role to retry on. `null` = top (the hero) or a leaf utility (embedder).
- **Shared endpoints** (`footprint_mb: 0, quant: "shared"`) mean the role reuses an already-loaded process (embedder/reranker on the hero) — zero extra RAM.

---

### 4.4 The routing policy (decision algorithm)

The router maps a **task** to a **route decision** = `{role, sampler_profile, grammar?, adapter?, spec_plan}`, then runs a **cascade with a confidence/entropy escalation gate**. The policy is *deterministic given inputs* (replayable, ch.02). It is **cheap by default** (a static task→role table) and *escalates only on measured uncertainty* — the unfair local advantage (free, exact confidence) instead of an external trained router.

#### 4.4.1 Inputs

```rust
struct RouteInput<'a> {
    task: TaskKind,                 // EditCode|GenerateCode|Plan|ToolCall|Chat|Summarize|Classify|Embed|Rerank|LongWatch|Explain
    prompt_meta: PromptMeta,        // token_len, has_multistep_markers, language?, retrieval_coverage, file_span_count
    caller_pref: CallerPolicy,      // ch.02 may force a role / determinism / max_latency / budget
    budget: BudgetState,            // §4.11: thermal_headroom, on_battery, ram_free_mb, concurrency
    registry: &'a RoleRegistry,
}
```

#### 4.4.2 The decision algorithm (pseudocode)

```python
def route(inp) -> RouteDecision:
    reg = inp.registry

    # ── 1. STATIC task → (role, sampler, grammar, adapter) base mapping ──
    #     Cheap, deterministic, the common case. No model call.
    base = TASK_TABLE[inp.task]        # e.g. EditCode → (hero, "edit", edit_format_grammar, lang_adapter)
    role     = base.role
    sampler  = inp.caller_pref.sampler or base.sampler
    grammar  = base.grammar(inp)       # may compile a per-tool / per-format grammar (§4.5)
    adapter  = pick_adapter(inp)       # §4.9: language/task adapter id, or None

    # ── 2. Architecture routing (transformer vs SSM), from ch.04's rule ──
    if inp.task == LongWatch or (inp.prompt_meta.token_len > reg[role].ctx_len and needs_streaming(inp)):
        role = "ssm_long"              # O(1)-state flat throughput; ch.04 §4.4.4
    # exact-recall multi-file edits NEVER route to SSM (recall weakness) — TASK_TABLE encodes this.

    # ── 3. DIFFICULTY-aware downgrade: try a cheaper role first when safe ──
    #     A cascade: cheap role, escalate UP on low confidence (step 5).
    if reg.routing.escalation != "off" and downgradable(inp.task) and inp.caller_pref.allow_cascade:
        cheaper = cheapest_role_meeting(inp.task, reg)   # e.g. Summarize → compactor; trivial Chat → fast_draft
        if cheaper and budget_allows(cheaper, inp.budget):
            role = cheaper             # we'll escalate to base.role if it stumbles

    # ── 4. BUDGET admission (energy/thermal/RAM) — may force a smaller role ──
    role = admit_under_budget(role, inp.budget, reg)     # §4.11; on-battery/hot → prefer smaller/local

    spec = plan_speculation(role, inp, reg)              # §4.8: draft role + draft_len, or NoSpec

    return RouteDecision(role, sampler, grammar, adapter, spec, cascade_to=reg[role].escalates_to)


# ── 5. EXECUTION with escalation (the cascade gate) — run by inference.rs ──
def execute(decision, inp) -> Result:
    out = call_role(decision, inp)                       # stream tokens + (gated) confidence signal

    # Escalation triggers — ALL use signals cloud can't give you for free:
    needs_escalation = (
        out.mean_token_confidence < THRESH[inp.task].conf          # §4.7 logit confidence
        or out.first_token_entropy > THRESH[inp.task].entropy      # high entropy ⇒ hard
        or out.grammar_dead_ended                                  # constraint hit an impossible state
        or (inp.task in VOTABLE and not self_consistent(out, k=3)) # §4.7 self-consistency disagreement
        or out.contains_low_logprob_span(tau=THRESH.span)          # a hallucination-suspect run
    )
    if needs_escalation and decision.cascade_to and budget_allows(decision.cascade_to, inp.budget):
        bigger = RouteDecision(role=decision.cascade_to, sampler="edit",
                               grammar=decision.grammar, adapter=decision.adapter,
                               spec=plan_speculation(decision.cascade_to, inp, inp.registry))
        out = call_role(bigger, inp)                     # escalate UP; record both in the manifest
        record_escalation(inp, from_=decision.role, to=bigger.role, reason=...)
    return out
```

**Key properties.**
- **Common case is free**: a static table picks the role; no extra model call to decide. (We don't pay a router-model tax on every turn.)
- **Escalation gates are *our* signals**: token confidence, first-token entropy, grammar-dead-end, self-consistency disagreement, low-logprob spans — *all exact, on-device, free to compute* (cost is the gated readback, §4.7). RouteLLM needs an offline-trained external router; we get a **cascade with a free, exact escalation oracle** out of the box, and add a learned router later (trained on accept/edit telemetry, ch.02).
- **Architecture-correct**: SSM vs transformer is a routing decision honoring ch.04's rule (exact-recall→transformer, unbounded-history→SSM).
- **Budget-aware**: on battery / thermally throttled, the router prefers smaller/local roles and less spec (§4.11).
- **Deterministic & replayable**: same inputs → same decision; the manifest records the role, sampler, grammar, adapter, and any escalation (ch.02 replay).
- **Caller override**: ch.02 can force determinism / a specific role / a latency cap (`CallerPolicy`); the router respects it.

**The `TASK_TABLE` (illustrative defaults):**

| TaskKind | Base role | Sampler | Grammar | Adapter | Cascade |
|---|---|---|---|---|---|
| `EditCode` | hero | **edit** (greedy+seed) | edit-format (§4.5) | language | — |
| `GenerateCode` | hero | edit | code fence / language | language | — |
| `Plan` | hero | balanced | plan-schema | — | — |
| `ToolCall` | hero (downgrade→fast_draft for simple tools) | **edit** | **per-tool arg schema** (§4.5) | — | →hero |
| `Chat` | fast_draft (downgrade) | balanced | — | — | →hero |
| `Summarize` | compactor | balanced | — | — | →hero |
| `Classify` | classifier | greedy | **enum** (fixed labels) | — | →hero |
| `Embed` | embedder | — | — | — | — |
| `Rerank` | reranker | edit | rank-list schema | — | →hero |
| `LongWatch` | ssm_long | balanced | — | — | — |
| `Explain` | hero | balanced | — | — | — |

---

### 4.5 Constrained / grammar decode as a first-class service

This is the **"small models become reliable" lever**, and HIDE's most decisive use of decoder ownership. The thesis: *don't ask a 7B model to please produce valid JSON — make every invalid token unreachable at decode time.* We already own the primitive (`json_constrain.rs::mask_logits` sets invalid-continuation tokens to `-inf` before sampling). The service generalizes it from hard-coded generic JSON to **any grammar** (JSON-Schema, GBNF, tool-call arg schema, plan schema, edit format, enum), compiled once and executed cheaply — the XGrammar pattern, reusing our mask machinery.

#### 4.5.1 The service interface (binding — [Appendix A.2])

```rust
/// Compiles a grammar spec to an executable mask program, once, cached.
trait GrammarService {
    /// Compile a grammar spec → a CompiledGrammar (the mask program), cached by
    /// (grammar_hash, vocab_sig). XGrammar-style: split vocab into
    /// context-independent (precomputed allow/deny) vs context-dependent.
    fn compile(&self, spec: &GrammarSpec, vocab: &VocabIndex) -> Arc<CompiledGrammar>;
}

enum GrammarSpec {
    JsonSchema(serde_json::Value),     // ch.03 tool args, plan schema, structured output
    Gbnf(String),                      // a BNF-like grammar (code/DSL)
    Regex(String),                     // identifiers, numbers, dates, fixed formats
    Enum(Vec<String>),                 // classifier labels / fixed choice
    ToolCall { tools: Vec<ToolSchema> }, // ch.03 registry → a union grammar over tool calls
    EditFormat(EditFormatSpec),        // HIDE's diff/patch emission format
}

/// The decode-time executor: one instance per active generation.
struct GrammarMatcher {
    compiled: Arc<CompiledGrammar>,
    state: GrammarState,               // PDA stack / FSM state (cf. json_constrain depth+state)
}
impl GrammarMatcher {
    /// Set invalid next tokens to -inf in `logits` for the CURRENT state.
    /// This IS json_constrain.rs::mask_logits, generalized to the compiled grammar.
    fn mask_logits(&self, logits: &mut [f32]);
    /// Advance the state by the chosen token's text (cf. json_constrain advance()).
    fn accept_token(&mut self, token_id: u32);
    /// Has the grammar reached an accepting end state? (cf. is_done())
    fn is_complete(&self) -> bool;
    /// True if the current state has NO valid next token (dead end → escalate, §4.4/§6).
    fn is_dead_ended(&self) -> bool;
}
```

#### 4.5.2 How a JSON-Schema / tool-call compiles to `mask_logits`

The path from "ch.03 hands the model layer a tool with an arg schema" to "the model *cannot* emit an invalid call":

1. **ch.03 → grammar.** The tool registry (ch.03) exposes each tool's argument JSON-Schema. `tool_call.rs` builds a *union grammar*: `tool_call ::= "{" '"name"' ":" tool_name_enum "," '"arguments"' ":" arg_object "}"`, where `tool_name_enum` is an `Enum` over the available tool names and `arg_object` is the selected tool's `JsonSchema` (the schema becomes active *after* the name is decoded — a context-dependent transition the PDA handles).
2. **Compile (once).** `compile()` lowers the schema to a state machine and, per state, precomputes the **context-independent** token partition (XGrammar's key trick): most of the 150K-token vocab is *unconditionally* allowed or denied in a given state (e.g. inside a string, only `"`-closing and escape transitions are context-dependent; the rest of the alphabet is a static allow-set). This makes the per-step mask a cheap lookup, not a vocab scan. Cached by `(grammar_hash, vocab_sig)` so the same tool-set never recompiles (`cache.rs`).
3. **Decode (per token).** Before each sample, `GrammarMatcher::mask_logits(&mut logits)` ORs the precomputed context-independent mask with the small context-dependent check, setting invalid tokens to `-inf`. The sampler (any profile) then draws from the *legal* tokens only. `accept_token` advances the PDA. `is_complete` signals the call is closed → stop.
4. **Result.** The emitted tool-call is **schema-valid by construction** — the model literally cannot produce `{"name": "edt_file"` (typo) or a missing required arg or a wrong-typed value, because those tokens were masked. This is the reliability jump: a 7B model with strict tool grammars matches the *format* reliability of a frontier model's strict mode, locally and for free. ([Dong et al. 2024], structured-output reliability work §3.2)

#### 4.5.3 What we grammar, and what we deliberately don't

- **Grammar:** tool-call args, plan schemas, classifier outputs (enum), structured extractions, the **edit/diff emission format** (so a patch is always apply-able), JSON responses. These are *format-bound* outputs where invalidity is pure loss.
- **Do NOT grammar:** the model's **reasoning / chain-of-thought**, free-form code *bodies* (beyond a fence/language hint), and natural-language explanation. The 2025 finding ([arXiv:2601.07525], JSONSchemaBench) is that **over-constraining hurts task quality when the grammar fights the model's natural token order**. So HIDE uses a **two-phase pattern**: a *think* phase (unconstrained), then an *emit* phase (constrained) — e.g. the model reasons freely about *which* tool and *what* args, then emits the tool-call under the grammar. The grammar guarantees the *envelope*, never the *thought*.

#### 4.5.4 Status & fallback

- **[RUNTIME-SIDE — LATER]** for the general grammar param: the runtime currently honors only `json_mode` (generic JSON). The ask (Appendix B #2) is a `grammar`/`response_format: {type: json_schema, schema}`/`tools` (strict) request field that selects a `CompiledGrammar`, plus generalizing `json_constrain`'s valid-set to be grammar-driven. **This is a high-leverage, medium-build runtime item** — the primitive (`mask_logits`) already exists; the work is the compiler front-end + the request plumbing + the XGrammar-style context-independent precompute.
- **[SHELL-TODAY]** fallback: (a) for JSON, use the existing `json_mode` flag end-to-end now; (b) for richer schemas, the shell does **constrained-via-validate-and-retry** — generate, parse against the schema, on failure re-prompt with the error (the cloud approach) — *plus* a cheap shell-side prefix-grammar where the tokenizer makes it tractable. It's slower and not *guaranteed*, but it works against today's surface while the runtime grammar param lands.

---

### 4.6 Sampler profiles

A **SamplerProfile** is a named, per-task sampler spec. The router attaches one to every request (the role's default, overridable by the caller). It is a **binding contract** — [Appendix A.3]. It *supersets* the in-tree `SamplingParams` additively (the sampler loop in `sample.rs` is already mask→penalize→truncate→draw; the new knobs slot in).

```rust
struct SamplerProfile {
    name: String,                      // "edit" | "balanced" | "explore" | "greedy" | custom
    temperature: f32,
    top_k: u32,
    top_p: f32,                        // (in-tree)
    min_p: f32,                        // [NEW] dynamic truncation = min_p * max_prob (§3.4); 0 = off
    typical_p: f32,                    // [NEW] locally-typical; 1.0 = off
    repetition_penalty: f32,           // (in-tree, flat)
    dry: Option<DryParams>,            // [NEW] sequence-aware repetition (n-gram extension penalty)
    logit_bias: Vec<(u32, f32)>,       // [NEW] per-token bias; +inf=force, -inf=ban (atop mask_logits)
    seed: Option<u64>,                 // (in-tree) — set for determinism
    stop: Vec<String>,
}
```

**The preset profiles:**

| Profile | temp | top_p | min_p | dry | seed | Use |
|---|---|---|---|---|---|---|
| **greedy** | 0.0 | — | — | — | fixed | Drafts, classification, anything argmax. Bit-reproducible. |
| **edit** | 0.0–0.2 | 0.95 | 0.05 | on | **fixed** | **Code edits, patches, tool-calls.** Determinism is the point; DRY kills loop-repeats without nuking legit code repetition; tiny temp only as a tie-breaker. |
| **balanced** | 0.7 | 0.9 | 0.05 | on | none | General chat, explanation, summarization (the current default's spirit). |
| **explore** | 1.0–1.2 | 0.98 | 0.1 | on | none | Brainstorming, naming, alternative-approach generation, search-query expansion. min-p keeps it coherent at high temp. |

**Design rationale.**
- **Determinism for edits is non-negotiable.** A patch must be reproducible (replay, ch.02) and stable (the user trusts it). `edit` is greedy+seeded; the grammar (edit-format, §4.5) does the *validity*, the sampler does the *reproducibility*.
- **DRY over flat rep-penalty for code.** Flat `repetition_penalty` punishes *all* recurrences — disastrous for code (which legitimately repeats `self.`, `let `, `}`). DRY penalizes only tokens that would *extend a repeated n-gram*, killing pathological loops (the model spiraling on boilerplate) without harming real code. This is a strict upgrade for our domain.
- **min-p as an *option*, default-conservative.** The literature is split (gains at high temp [Nguyen et al. 2024] vs a human-eval null [arXiv:2506.13681]). We ship it as a knob, default it on lightly in `explore`/`edit` (where it helps coherence at the chosen temp), and **A/B it locally on our own quality oracle** before making it a default — exactly the in-tree "measure before flipping" discipline.
- **logit-bias is a free safety/format lever.** Atop `mask_logits`: ban `eval(`/`exec(` in generated code by policy, forbid a deprecated API the project memory flags, force a fence language. Trivial, powerful, decoder-only.

**Status.** **[RUNTIME-SIDE — LATER]** for the superset fields (additive to `SamplingParams` + the sampler loop; Appendix B #4). **[SHELL-TODAY]:** today's temp/top_k/top_p/seed cover `greedy`/`balanced`/`edit`(minus DRY/min-p) right now over `/v1/chat/completions` and `/v1/hawking/generate`. We get determinism (seed) and low-temp edits *today*; DRY/min-p/typical/logit-bias are the additive ask.

---

### 4.7 Logit-level features exposed to the agent

The decoder's distribution is an **API surface** for the agent (ch.02) and the router (§4.4). These are the signals cloud can't give you exactly or for free. **The cost model matters:** the Metal sampler keeps logits on-GPU and reads back **4 bytes** (the argmax id); computing confidence/entropy/logprobs requires a **full or top-k logit readback** (`vocab×4` or `k×4` bytes/step). So these features are a **deliberate, gated `want_logprobs` path** — off by default (cheap greedy), on when the agent asks (escalation steps, voting, hallucination checks). This is already visible in `GenStats` (`readback_bytes`, `logits_materialized_rows`, `token_only_path_used`) — the runtime *tracks* the readback choice; the ask is to *expose* the values (Appendix B #1).

The exposed signals (binding shapes in [Appendix A.4]):

| Signal | Definition | What the agent does with it |
|---|---|---|
| **token confidence** | `softmax(logits)[chosen]` per token; or the **margin** `p1 − p2` (top-2 gap). | Flag low-confidence spans (likely-wrong API names, uncertain edits); **escalate** to the hero (§4.4); annotate diffs the user should double-check. |
| **entropy** | `H = −Σ p log p` over the (top-k) distribution per step; **first-token entropy** especially. | **Difficulty estimate** — high first-token entropy predicts a hard generation → route up or spend more compute. Entropy-gate: only invoke expensive escalation when entropy says it's worth it. |
| **self-certainty** | divergence-from-uniform of the full distribution ([arXiv:2502.18581]); separates correct/incorrect better than perplexity. | **Best-of-N selection** and **early-stopping** by confidence; rank N sampled candidates without a separate verifier. |
| **n-best logprobs** | top-k token logprobs per position. | Surface *alternatives* (e.g. "did you mean `parse_u64`?"), feed constrained-decode tie-breaks, power logprob-guided search. |
| **self-consistency** | sample N completions (cheap profile), majority-vote / cluster the *answer* (the tool-call, the chosen value, the diff). | For `VOTABLE` tasks (a classification, a single-value extraction, a yes/no plan gate): **vote**; disagreement → escalate. ([Wang et al. 2023b]) |
| **logprob-guided best-of-N** | generate N (with `explore`), score each by self-certainty/verifier, pick the best. | Quality lever for hard generations on a cheap model — test-time compute *we* control, no per-sample bill. |

**The escalation/voting ladder (how §4.4 step 5 consumes these):**

```
run cheap role (greedy/balanced, NO logprob readback) ──► answer + 4-byte argmax stream
        │
        │ if task is gateable (tool-call / classify / single-value / risky edit):
        ▼
  request a CONFIDENCE pass (gated readback ON for this gen):
     mean_token_confidence, first_token_entropy, low-logprob spans
        │
        ├─ confident & grammar-complete & no low-logprob span ──► ACCEPT
        ├─ uncertain & VOTABLE ──► self-consistency (N=3 cheap samples) ──► agree? accept : escalate
        └─ uncertain / grammar-dead-end / hallucination-suspect ──► ESCALATE to escalates_to (hero)
```

**Why cloud can't:** OpenAI/Anthropic expose at most truncated top-k logprobs (often off by default), no full distribution, no per-layer hidden states, and *never* let you run "N cheap samples then vote" without N× billing. **We compute the exact distribution on-device; N concurrent cheap samples cost only idle silicon.** Self-certainty escalation, entropy-gated routing, and free best-of-N are *native* HIDE capabilities. This is the chapter's sharpest "cloud literally cannot do this."

**Status.** **[RUNTIME-SIDE — LATER]** to expose logprobs/entropy/confidence on the API (Appendix B #1 — the *highest-leverage* ask, because it unlocks §4.4 escalation, §4.7 voting, and hallucination detection). **[SHELL-TODAY]:** self-consistency *voting* works now (N requests over `/v1/chat/completions`, vote shell-side) — no new hook needed, just N calls; the *per-token* confidence/entropy signals wait on the readback exposure.

---

### 4.8 Speculative decode strategy

Spec-decode is **free latency**: a cheap draft proposes K tokens, the hero verifies them in one batched forward, and the verified-greedy output is **bit-identical to no-spec** (the losslessness guarantee on the ExactShared/Eagle5 path). HIDE already ships the *fleet and the arbiter* (`speculate/`); this section specifies *strategy* — which proposer, when, and why agent loops are the ideal case.

#### 4.8.1 The in-tree machinery (what we build on)

- **Proposers** (`ProposerId`): `UserNgram` (user's own recent tokens), `SuffixArray`/`suffix_automaton` (corpus/context retrieval drafts, REST-style), `Eagle5` (trained neural head — mock today), `CrossTokenizer` (a *different-family* small model drafting for the hero), `Retrieval`, `Tree` (token-tree), `ParallelDraft`.
- **Wall-clock router** (`router.rs`): a measured `verify_cost_forwards(b)` curve (B=1→1.0, B=8→4.15) and per-proposer EWMA cost models (`draft_ns`, `verify_extra_ns`, `retokenize_ns`, `sync_ns`, `hit_frac`). It only proposes when `expected_accepted_tokens × value_per_token > verify_cost` — i.e. **it auto-kills spec when it doesn't pay** (e.g. on a fast small model where a saved token is worth little).
- **Governor** (`spec_gov.rs` / `governor.rs`): rolling accept-rate auto-disable (min 0.35, 5 consecutive-reject ceiling) + a **UCB1 bandit** (`policy.rs`) for proposer selection under uncertainty.

#### 4.8.2 The strategy

| Situation | Plan | Rationale |
|---|---|---|
| **Hero generating code/edits** (low-temp/greedy) | **Eagle5 (trained) draft → hero verify**, or **fast-draft (0.5B) cross-tokenizer draft → hero verify**. draft_len from `context_confidence`. | Code is *predictable* (high local n-gram structure) → high accept length → spec clears the cost curve handily. Greedy verify → lossless. **This is the bread-and-butter win.** |
| **Hero in an agent loop re-emitting structured output** (tool-calls, diffs, plans) | **n-gram + suffix drafting** (the structure repeats) **under the grammar mask**. | Agent loops re-emit boilerplate (the same JSON envelope, the same imports, the same diff scaffolding). Draft tokens are *cheap to predict* and *the grammar already constrains them* — accept rates are high. **Agent loops are the ideal spec case** because the output is structured and self-similar across turns. |
| **Fast-draft as a standalone responder** | **No spec** (it's already cheap; verify cost > value). | The router's `target_ns_per_token` is small on a 0.5B → spec auto-disables. Don't draft a draft. |
| **SSM long-context** | **No spec** (recurrent, the cost model differs; flat decode already). | SSM decode is already O(1)/flat; spec's batched-verify assumption doesn't map cleanly. Leave it. |
| **High-temperature explore** | **Spec off or speculative-sampling (distribution-preserving) only.** | At high temp, greedy-verify acceptance falls and the lossless guarantee needs the rejection-sampling variant; the router's accept-rate gate handles the downgrade. |

#### 4.8.3 Self-speculation (the no-extra-memory option)

When RAM is tight (no room for a second model) or thermals are constrained (§4.11), **self-speculation** — the hero drafts *itself* via layer-skip/early-exit, then verifies with full depth ([Elhoushi et al. 2024], [Zhang et al. 2023]) — gives ~1.3–1.8× with **zero extra parameters**. This is the **on-battery / low-RAM spec mode**: the scheduler (§4.11) selects self-spec over a separate draft when footprint matters. *Status:* the Eagle5 head is the in-tree neural-draft seam; self-spec/early-exit is a [RUNTIME-SIDE — LATER] add to the forward pass (a layer-subset forward + verify), naturally composing with the existing router.

#### 4.8.4 The honest caveats (carried from memory)

- **Trained head needed for neural spec.** Eagle5's mock head validates the *path*; real accept rate needs training (the EAGLE-3 recipe: direct-token draft + training-time-test on the hero's own rollouts, ~hundreds of K examples). This is a *Hawking Condense* training job → ties to §4.10.
- **Batched-verify ≠ greedy kernel at near-ties.** The EH property gate caught that batched verify can diverge from the single-token greedy kernel at near-ties → **non-lossless at the margin**. The lossless guarantee is enforced on the ExactShared/Eagle5 greedy-verify path; any tree/parallel-draft mode that risks it stays **gated behind the property test** (don't ship a mode that fails it). For correctness-critical edits, the safe default is greedy-verify spec or no spec.
- **Spec is a *latency* lever, never a quality lever.** It never changes *what* the model would have said (on the lossless path); it only makes it *arrive faster*. Quality comes from the model, the grammar, and the sampler — not from spec.

**Status.** **[PROVEN-IN-PROD in-tree]** for the fleet + router + governor + bandit (more sophisticated than most published systems). **Gaps:** a *trained* Eagle5/EAGLE-3 head (§4.10), the cross-tokenizer 0.5B→7B path validated, self-spec for low-RAM. All compose with the existing arbiter; no architecture change.

---

### 4.9 LoRA / adapter hot-swap

LoRA adapters let one base model wear many hats cheaply: a **per-language** adapter (`rust.lora`, `ts.lora`, `python.lora`) that knows the idioms; a **per-task** adapter (`commit-msg.lora`, `sql.lora`, `test-gen.lora`); and the **personal** adapter (§4.10) trained on *this user's* accepted edits. The base-model forward is shared; the adapter is a small low-rank delta — so swapping is cheap and several adapters can be resident at once (S-LoRA's Unified Paging).

#### 4.9.1 The selection seam (binding — [Appendix A.5])

```rust
/// Which adapter(s) to apply for a request. Resolved by the router (§4.4) from
/// task + detected language + user/project config, validated against role caps.
struct AdapterSelection {
    base_role: String,                 // e.g. "hero"
    adapters: Vec<AdapterRef>,         // usually 0..2: e.g. [language, personal]
}
struct AdapterRef { id: String, scale: f32 }   // scale = blend weight (composition)
```

The router picks the adapter (`pick_adapter` in §4.4.2): `EditCode` in a `.rs` file → `["rust", "personal"]`; `commit-msg` task → `["commit-msg"]`; unknown → none (base). The selection is recorded in the manifest (replay) and validated against the role's `caps.adapters`.

#### 4.9.2 Serving model

- **Resident set + on-demand fetch (S-LoRA pattern).** Keep the handful of *hot* adapters resident; fetch a cold one from disk on first use. On a single-user box the working set is tiny (current language + personal + maybe one task adapter) → near-zero overhead.
- **Composition.** Blend a language adapter + the personal adapter (`scale`-weighted) so the model writes idiomatic-language code *in the user's style*. (Compose carefully — adapter interference is real; gate compositions on the quality oracle.)
- **Shared base forward.** Multiple requests on different adapters share the base GEMM, applying their low-rank deltas (Punica/SGMV-style) — the multi-tenant efficiency, even single-user (the personal adapter on every request + a transient task adapter).

#### 4.9.3 Status & fallback

- **[RUNTIME-SIDE — LATER]** — **no LoRA serving exists in-tree today.** This is a clean build: an adapter loader, the low-rank delta in the forward (Metal), the resident-pool manager, and an `adapter` request field (Appendix B #3). Medium-high effort; high payoff for code quality (a Rust-tuned 7B writes much better Rust).
- **[SHELL-TODAY]** fallback: serve a *separately-loaded merged model* per heavily-used adapter as its own **role** (e.g. a `rust-hero` endpoint = base+rust merged), routed normally. Works on today's surface; costs the full RAM of each merged variant (so only for the 1–2 most-used). The personal adapter starts here (a periodically-merged personal hero) until live LoRA lands.

---

### 4.10 The fine-tune-at-condense personalization flywheel

This is the flywheel **cloud cannot offer**: because *Hawking Condense* makes the weights, HIDE can **train the local model on the user's own accepted edits and HIDE's exact tool protocol**, producing a model that gets *measurably better at this user's code in this user's style over time* — privately, on-device, with no per-token bill and no data leaving the machine. A cloud model is a frozen stranger; ours is a colleague that learns you.

#### 4.10.1 The loop

```
   (1) CAPTURE          (2) CURATE              (3) TRAIN (Condense)      (4) DEPLOY
   accepted diffs   →   build a clean        →  fine-tune a LoRA      →  hot-swap the
   tool-call traces     preference dataset      (or fold into .tq        "personal" adapter
   user corrections     (accept = positive,     condense) on the         / new role artifact
   chosen completions   reject/undo = neg)      dataset                   → router uses it
        ▲                                                                      │
        └──────────────────────── more, better-aligned generations ───────────┘
```

1. **Capture (shell-side, ch.02 telemetry).** Every *accepted* edit (the diff the user kept), every *rejected/undone* suggestion, every successful tool-call, and every user correction is logged with provenance (the same accept/edit signal ch.02 produces and ch.04 §9 wants for scoring weights). **This is local, private, opt-in, user-inspectable data** — the user's own work.
2. **Curate.** Build a **preference / SFT dataset**: accepted diffs as positive targets (input = the context the compiler produced, output = the edit the user kept); rejected suggestions as negatives (for DPO/KTO-style preference training); tool-call traces teaching HIDE's protocol; style exemplars (the user's formatting, naming, comment density). De-dup, filter secrets, cap size. The dataset is a HIDE artifact (`.hide/personal/dataset/`), versioned, user-deletable.
3. **Train (Condense's job).** *Hawking Condense* runs the fine-tune — **a LoRA on the hero** (cheap, hot-swappable, §4.9) is the default; folding into a re-condensed `.tq` is the heavier option. Two distinct objectives:
   - **Protocol fine-tune (ship-time, all users):** teach the model HIDE's tool-call format, edit format, and plan schema so it's *natively fluent* in our protocol (less reliance on grammar masking → fewer dead-ends, higher accept length for spec). This is a *Hawking-native* model, co-designed with the harness.
   - **Personal fine-tune (per-user, opt-in):** the `personal` adapter on the user's accepted-edit dataset → the model writes *this user's* style.
4. **Deploy.** The new adapter/checkpoint becomes a role artifact (§4.3); the router hot-swaps it in (§4.9). Telemetry on the *new* model's accept rate measures whether the flywheel helped (gate: personal adapter ships only if it raises accept rate on a held-out slice).

#### 4.10.2 Why this is the unfair advantage

- **Cloud can't:** a cloud model is shared and frozen; it cannot be fine-tuned on *your* private diffs (data-residency, cost, and they won't ship you weights). Even "memory" features are prompt-stuffing, not weight updates.
- **We own every piece:** the runtime (serves the adapter), the weights (Condense fine-tunes them), the data (the user's local accepted edits), and the loop (ch.02 telemetry → Condense → role registry). The whole flywheel is in-house.
- **Compounding:** the more you use HIDE, the more aligned the local model becomes to your code and style — a moat that *grows with usage* and is *non-transferable* (it's your model now).
- **Private by construction:** training is on-device (or on the user's own machine via Condense); the dataset never leaves. This is a *privacy feature*, not just a quality one.

#### 4.10.3 Status

- **Capture + curate: [SHELL-TODAY]-ish** — ch.02 produces the accept/edit telemetry; building the dataset is shell work (no runtime hook). Start here immediately (it's also ch.04 §9's scoring-weight data).
- **Train: roadmap (Condense tooling)** — the trainer (`tools/training/`, cf. the existing `eagle5_train.py`) is *Hawking Condense*'s deliverable; this chapter specifies the **dataset contract** (Appendix A.6) and the deploy seam (role artifact + adapter hot-swap).
- **Deploy: gated on §4.9** (LoRA serving) for live hot-swap; the **[SHELL-TODAY]** path is a periodically-merged personal model served as a role.

---

### 4.11 Multi-model concurrency & scheduling

On a single local box, **idle silicon is free** — running a hero + a fast-draft + an embedder concurrently costs only *power*, not dollars (the cloud charges per model per token; we pay watts). But power, heat, and RAM are *real* finite budgets on a laptop, so the scheduler treats them as first-class admission inputs.

#### 4.11.1 Concurrency model

- **One process per loaded model** (ground truth) → the **subprocess router** (ch.01) supervises N `hawking-serve` instances (hero:8081, fast-draft:8082, compactor:8083, …). Roles with `footprint_mb: 0, quant: "shared"` (embedder/reranker) reuse a process → no extra RAM.
- **The fleet runs concurrently at zero marginal *cost*:** the embedder embeds while the hero generates while the draft drafts. The only contention is the GPU/ANE and memory bandwidth — managed by the scheduler.

#### 4.11.2 Energy/thermal/RAM-aware admission (the laptop reality)

```python
def admit_under_budget(role, budget, reg) -> role:
    # RAM: don't load a role whose footprint won't fit alongside resident set.
    if reg[role].footprint_mb > budget.ram_free_mb:
        role = smaller_role_for(reg[role].role_kind, reg, fits=budget.ram_free_mb)  # or evict an idle role

    # THERMAL: if hot / throttled, prefer smaller roles and disable aggressive spec.
    if budget.thermal_headroom < THERMAL_LOW:
        role = prefer_smaller(role, reg)
        budget.spec_aggressiveness = "conservative"   # router proposes shorter drafts / self-spec only

    # BATTERY: on battery, bias to efficiency — smaller roles, self-spec, lower concurrency.
    if budget.on_battery and budget.mode != "plugged_perf":
        role = prefer_smaller(role, reg)
        budget.max_concurrency = min(budget.max_concurrency, 2)
    return role
```

- **User-facing modes** (the dial, mirroring ch.04's profile dial): **`plugged-perf`** (full fleet, full spec, biggest roles), **`balanced`** (default), **`quiet`/`on-battery`** (smaller roles, self-spec, throttled concurrency — quieter fans, longer battery). The router's `BudgetState` carries these; energy estimates (`energy_j_per_tok_est`) come from the in-tree per-domain J/tok measurement (the "genuine" energy verdict from memory).
- **Apple Silicon specifics:** route the embedder/classifier to the ANE/efficiency cores where the runtime supports it; keep the hero on the GPU; the unified-memory architecture means "footprint" is one shared pool (RAM = VRAM) — the scheduler budgets *one* number.
- **Thermal as a feature:** "HIDE got quieter and cooler when I unplugged" is a *user-delightful* behavior a cloud agent (which has no idea you're on battery) cannot produce.

**Status.** **[SHELL-TODAY]** for the policy (the router reads battery/thermal via OS APIs and picks roles/concurrency — pure shell). **[RUNTIME-SIDE — LATER]** for ANE routing and precise per-role energy accounting (the runtime exposes some via `GenStats`/`/metrics`; finer control is an ask).

---

## 5. How we EXCEED — "cloud literally cannot do this"

Each item is a capability that exists *only* because we own the runtime + weights + quantizer. Tagged with the mechanism and the cloud's hard limit.

1. **Grammar-valid tool-calls by construction.** We mask invalid tokens to `-inf` at decode time (§4.5, atop `json_constrain.rs`), so a small model *cannot* emit a malformed tool-call/plan/diff. **Cloud:** at best strict-mode JSON on *their* schema; they will not run *your* arbitrary GBNF/edit-format grammar against the decoder, and you can't reach the mask. **Result:** a 7B local model matches frontier *format* reliability — the "small models become reliable" lever.

2. **Exact, free logit confidence + entropy + self-certainty.** We compute the true distribution on-device (§4.7) → token confidence, first-token entropy (difficulty), self-certainty (correct/incorrect separation), low-logprob hallucination spans. **Cloud:** truncated top-k logprobs at best, often disabled; never the full distribution, never hidden states. **Result:** escalation, voting, and hallucination detection with a *free exact gate* the cloud can't expose.

3. **Self-consistency & best-of-N at zero marginal cost.** N concurrent cheap samples → vote / confidence-rank (§4.7) cost only idle silicon. **Cloud:** N× the bill, every time. **Result:** test-time compute *we* dial without a meter.

4. **Custom samplers per task.** Deterministic seeded greedy + DRY + edit-format grammar for patches; min-p/explore for brainstorming (§4.6). **Cloud:** a fixed temp/top-p knob set, no DRY, no per-token logit-bias, no determinism guarantee across their backend. **Result:** reproducible edits and coherent exploration, dialed to the task.

5. **LoRA hot-swap per language/task.** A Rust-tuned, then *user*-tuned 7B writes better Rust in *your* style (§4.9). **Cloud:** one frozen shared model; no per-user weights. **Result:** specialization the cloud structurally cannot ship to one user.

6. **Personalize the *model*, privately, forever (the flywheel).** Fine-tune on the user's accepted diffs and HIDE's protocol at condense-time (§4.10); the model compounds toward *your* code. **Cloud:** prompt-stuffed "memory" at most; your private diffs never become weights and never leave their datacenter on your terms. **Result:** a growing, non-transferable, private moat — *your* model.

7. **Speculative decode we own end-to-end.** The hero verifies a draft for free latency, losslessly, with a wall-clock arbiter that auto-disables when it stops paying (§4.8). **Cloud:** spec is their internal optimization; you can't choose the draft, tune accept thresholds, or use *your* corpus/n-grams as the draft source. **Result:** latency tuned to *our* code workload (where draft accept is high), with our own proposers.

8. **Architecture routing (transformer ↔ SSM).** Unbounded-history tasks go to RWKV-7/Mamba-2 (O(1) state, flat decode); exact-recall to the transformer (§4.4, ch.04). **Cloud:** one model architecture per endpoint, chosen by them. **Result:** the right *substrate* per task, a structural long-context win.

9. **Energy/thermal-aware scheduling.** The fleet shrinks and quiets on battery (§4.11). **Cloud:** no concept of your laptop's thermals. **Result:** a calmer, longer-lasting machine — a local-only delight.

10. **Multi-model concurrency for free.** Hero + draft + embedder + compactor at once, costing watts not dollars (§4.11). **Cloud:** every concurrent model multiplies the bill. **Result:** a rich fleet a metered budget would forbid.

---

## 6. Failure modes & mitigations

| # | Failure | Symptom | Mitigation (mechanism) |
|---|---|---|---|
| F1 | **Mis-routing** (wrong role for the task) | A hard task sent to a too-small role → bad output; or an easy task wasting the hero. | **Cascade + free escalation gate** (§4.4): cheap role runs, *measured* low confidence/high entropy/grammar-dead-end/self-consistency-disagreement escalates *up*. The static table is conservative; escalation is the safety net. Telemetry tunes the table; a learned router (RouteLLM-style) is the upgrade (§8). |
| F2 | **Draft mismatch** (spec accept rate collapses) | Spec proposes tokens the hero rejects → *slower* than no spec. | **Wall-clock router + governor already handle it** (§4.8): `verify_cost_forwards` + EWMA cost models propose only when expected accept clears the cost; the governor auto-disables after 5 consecutive rejects / accept<0.35; the UCB1 bandit reallocates to a better proposer. Spec can only *help or no-op*, never silently hurt for long. |
| F3 | **Over-constrained decode** (grammar fights the model) | Constrained output is *valid but dumb*; or reasoning degrades because CoT was grammared. | **Two-phase think→emit** (§4.5.3): never grammar the chain-of-thought or free code bodies; grammar only the *envelope* (tool-call args, diff format, enum). Detect **grammar dead-ends** (`is_dead_ended`) → the constraint painted the model into a corner → back off / escalate / widen the grammar. Gate constraints on the task-quality oracle (constraints that *lower* accept rate get relaxed). |
| F4 | **Grammar dead-end** (no valid next token) | Generation stuck; mask zeroes everything. | `GrammarMatcher::is_dead_ended()` is checked each step; on dead-end: (a) if the grammar was over-tight, fall back to validate-and-retry for that span; (b) escalate to the hero; (c) surface to the agent (ch.02) as a tool-arg error to re-plan. Never deadlock (the in-tree mask already allows empty/BOS tokens to avoid hard stalls). |
| F5 | **Spec non-lossless at near-ties** (the EH finding) | Batched verify diverges from greedy kernel at a tie → output differs from no-spec. | **Gate risky modes behind the EH property test** (§4.8.4): ship only ExactShared/Eagle5 greedy-verify (proven lossless) for correctness-critical edits; tree/parallel-draft modes stay opt-in/gated; for the `edit` profile, default to greedy-verify spec or no spec. *Held items stay held until the property gate is green.* |
| F6 | **Confidence-readback cost** (logprobs are not free on GPU) | Turning on `want_logprobs` for every token tanks tps (full `vocab×4` readback/step vs 4 bytes). | **Gated, on-demand readback** (§4.7): default greedy 4-byte path; full/top-k readback only on the *confidence pass* (escalation, voting) or top-k (not full vocab) when a sample needs alternatives. `GenStats` already tracks `readback_bytes`/`token_only_path_used` so the cost is *visible* and budgetable. |
| F7 | **LoRA interference** (composed adapters hurt) | language+personal blend degrades quality vs either alone. | **Gate compositions on the quality oracle** (§4.9): a composition ships only if it beats the better single adapter on a held-out slice; default to *one* adapter when in doubt; `scale` blends conservatively. The personal adapter ships only if it raises accept rate (§4.10). |
| F8 | **Personalization overfit / drift** (the flywheel learns bad habits) | The model amplifies a user's mistake, or a rejected pattern leaks in as positive. | **Curate strictly** (§4.10.2): only *accepted* edits are positive; rejected/undone are negatives (DPO-style); held-out accept-rate gate before deploy; the dataset is **user-inspectable and deletable** (it's their files); versioned adapters so a bad personalization rolls back. Secret-filtering on the dataset. |
| F9 | **Architecture recall miss** (SSM route loses a needle) | An unbounded-history task on RWKV-7 misses an exact long-range fact. | **Route by task** (§4.4, ch.04 F9): exact-recall tasks never go to SSM; the dial documents the tradeoff; hybrid models (§8) recover recall. |
| F10 | **Role process down / endpoint unreachable** | A served instance crashed; the router points at a dead endpoint. | The **subprocess router** (ch.01) supervises/restarts; the registry's `escalates_to` and shared-endpoint fallbacks give a degraded path (e.g. fast-draft down → route its tasks straight to hero); health via `/healthz`. |
| F11 | **Determinism break** (an edit isn't reproducible) | Replay (ch.02) produces a different patch. | The `edit`/`greedy` profiles are **seeded**; greedy at temp 0 is argmax; spec on the greedy-verify path is bit-identical; the grammar is deterministic; the route decision is recorded in the manifest. Same inputs → same patch. |
| F12 | **Tokenizer/vocab skew in the grammar** (mask built for the wrong vocab) | A compiled grammar masks the wrong tokens after a model swap. | `CompiledGrammar` is cached by `(grammar_hash, **vocab_sig**)`; a vocab/tokenizer change invalidates the cache; the role's `tokenizer_sig` (in `ModelDescriptor`) is checked — same discipline as the in-tree prefix-cache keying. |

---

## 7. Extensibility / plugin points

Everything that *is a model*, *shapes a decode*, or *decides a route* is a trait/registry entry, so new capability drops in without forking.

1. **`RoleRegistry` entry** (§4.3) — add a new role (a new model for an existing kind, or a new kind) by adding a descriptor. This is ch.01's `ModelProvider`/`ProviderCaps` seam, role-specialized; ch.05's "new embedding / reranker role" plugs in exactly here.
2. **`GrammarSpec` variant** (§4.5) — add a new grammar source (a new DSL, a new structured format, a domain schema) by implementing a compiler to `CompiledGrammar`; it becomes a `grammar` the router can attach.
3. **`SamplerProfile`** (§4.6) — ship/share new sampler presets per language/team as files under `.hide/profiles/`; add a new sampler *primitive* (a future truncation method) additively to the sampler loop.
4. **`ProposerId`** (§4.8) — add a new spec proposer behind the in-tree router (the fleet already has 8); the wall-clock arbiter + bandit pick it up automatically once its cost model is registered.
5. **Adapter** (§4.9) — register a LoRA (`id`, base role, training provenance); the router's `pick_adapter` can select it; compositions are gated.
6. **Difficulty estimator / escalation signal** (§4.4) — plug a new difficulty proxy or escalation trigger (e.g. a learned difficulty classifier, a domain verifier) into the cascade gate.
7. **Router policy** — swap the static `TASK_TABLE` + escalation for a *learned* router (RouteLLM-style, trained on accept/edit telemetry) behind the same `route()` signature.
8. **Confidence aggregator** (§4.7) — add a new test-time-compute strategy (a verifier ensemble, a new self-certainty metric) behind the confidence interface.

Stability contract: `RoleDescriptor`, `GrammarSpec`/`CompiledGrammar` interface, `SamplerProfile`, `AdapterSelection`, and the logit-signal shapes are **versioned** (`schema_version`); additive = minor, breaking = major; consumers pin a major.

---

## 8. Bleeding-edge / moonshots (ranked)

Ranked by **(impact × feasibility) ÷ cost**, each tagged maturity / difficulty.

1. **General grammar/constrained-decode service (JSON-Schema/GBNF/tool-call → `mask_logits`).** *Highest leverage:* the primitive exists (`json_constrain.rs`); generalizing it to schema-driven masking + the request hook is the single biggest reliability jump for small models, and the field is fully solved (XGrammar). **[RESEARCH-PROVEN tech / med build]**. **Do this first** (Appendix B #2). Add the XGrammar context-independent/dependent split for cheapness.
2. **Expose logprobs/entropy/confidence on the API (gated readback).** *Unlocks §4.4 escalation, §4.7 voting, hallucination detection* — the whole "logits as an API" pillar. Cheap-ish (the sampler already tracks readback choice). **[PROVEN tech / med build]**. (Appendix B #1).
3. **The personalization flywheel, capture+curate first.** Start logging accepted-edit datasets *now* (also feeds ch.04 §9 scoring weights); the trainer is Condense's. Biggest *user-felt, cloud-impossible* differentiator; lowest near-term risk (capture is pure shell). **[low build for capture / very high impact]**. *Do early.*
4. **Train a real Eagle5/EAGLE-3 head** on the hero's own rollouts (training-time-test recipe) for 3–6× spec on code. Composes with the in-tree router. **[RESEARCH-PROVEN / high build — a Condense training job]**.
5. **LoRA serving + hot-swap (S-LoRA pattern).** Per-language + personal adapters; a Rust-tuned-then-you-tuned 7B. High code-quality payoff. **[PROVEN-IN-PROD elsewhere / high build]** (Appendix B #3). The personal adapter is the flywheel's deploy target.
6. **Sampler superset (min-p/typical/DRY/logit-bias) + the `edit` profile.** DRY for code, determinism for edits — strict quality-of-life wins; small additive build. **[RESEARCH-PROVEN / low–med build]** (Appendix B #4).
7. **Protocol fine-tune (ship-time):** a Hawking-native hero fluent in HIDE's tool/edit/plan format → fewer grammar dead-ends, higher spec accept. **[RESEARCH-PROVEN / med–high build — Condense]**. The model and harness co-designed.
8. **Learned router** (RouteLLM matrix-factorization / classifier) trained on our accept/edit telemetry → smarter task→role than the static table. **[PROVEN-IN-PROD / med build]** once telemetry exists.
9. **Self-speculation (layer-skip/early-exit)** as the low-RAM/on-battery spec mode — zero extra parameters, ~1.3–1.8×. **[RESEARCH-PROVEN / med build]**.
10. **Hybrid attention–SSM hero** (a few attention layers among RWKV-7/Mamba-2 layers) to get flat throughput *and* exact recall — resolves the F9 routing tradeoff. Ties to ch.04 §8 + Condense. **[SPECULATIVE→RESEARCH-PROVEN / high build]**.
11. **Verifier-ensemble best-of-N** (multiple heterogeneous local verifiers scoring N candidates) as a test-time-compute axis for hard generations. **[RESEARCH-PROVEN / med build]**.
12. **Activation steering hooks** (control vectors for conciseness / style / refusal) — *opt-in, gated, off the correctness path* given the 2026 reliability caveats. We own the residual stream, so it's *possible*; treat as research. **[SPECULATIVE / med build]**.

---

## 9. Open questions & dials

- **Escalation thresholds.** What token-confidence / first-token-entropy / self-consistency-disagreement levels should trigger escalation per task? Defaults from the self-certainty literature; **tune on our own accept/edit telemetry** (the gate must not over-escalate to the hero, killing the cost savings, nor under-escalate, shipping bad output). Dial: `THRESH[task]`.
- **Static table vs learned router — when to switch.** The static `TASK_TABLE` + free escalation gets us 90% of RouteLLM's value with zero training; when does the learned router pay for itself? (Probably once we have enough accept/edit data and a stable role fleet.)
- **Grammar vs quality tradeoff.** Which outputs *benefit* from a grammar and which are *hurt* (the over-constraining finding)? The think→emit split is the rule, but where exactly is the boundary for, e.g., a code body vs a tool-call? Dial: per-task `grammar` on/off + the think/emit phase boundary.
- **min-p / sampler defaults.** Ship min-p on or off by default? The literature is split — **A/B on our quality oracle** before defaulting. Per-profile dial.
- **Logprob readback budget.** How much of the time can we afford the full/top-k readback (for confidence/voting) before it costs noticeable tps? Dial: `want_logprobs` policy (always / on-escalation / top-k-only).
- **Spec proposer choice for code.** Trained Eagle5 vs cross-tokenizer 0.5B vs n-gram/suffix — which wins on *our* code workload, and does the bandit converge fast enough? (The router measures it; the question is the prior.)
- **LoRA composition.** How many adapters can compose (language + personal + task) before interference? Default 1–2; gate the rest.
- **Personalization data policy.** What's positive vs negative in the dataset (accepted edit = positive; but is a *modified-then-accepted* suggestion a positive for the original or the modification)? Decay of old style? Secret-filtering rules. The dataset is the user's — what are the inspection/deletion/export affordances?
- **Self vs separate-model spec under RAM pressure.** The crossover where self-spec (no extra RAM, lower accept) beats a separate draft (extra RAM, higher accept). Scheduler dial.
- **Energy/thermal mode defaults.** Where are the `plugged-perf` / `balanced` / `quiet` boundaries (thermal headroom %, battery %)? User-overridable, but the defaults matter.
- **Which roles to actually run by default.** Hero alone? Hero + draft? The full six? On an 18GB Mac the footprint budget (§4.11) caps it — what's the default fleet per machine tier?

---

## 10. Cross-references

- **ch.01 · System architecture.** Owns the **subprocess model router** (the process that spawns/supervises N `hawking-serve` instances) and the **`ModelProvider`/`ProviderCaps`** plugin seam. This chapter *specifies the roles and the routing policy* that process runs; our `RoleDescriptor.caps` *is* `ProviderCaps` role-specialized. **ch.01 binds the role registry + route decision as the policy its router executes.**
- **ch.02 · Agent loop.** Consumes this chapter's three pillars: **logit-confidence** (escalation/hallucination gates, §4.7), **grammar-constrained tool-calls** (reliable tool use, §4.5), and **speculative self-drafting hooks** (latency, §4.8). The loop's **accept/edit telemetry** is the fuel for the personalization flywheel (§4.10) and the learned router (§8). The route decision + sampler + grammar are recorded for **replay** (ch.02's determinism). **ch.02 binds `InferenceRequest`/the route decision and the logit-signal shapes (A.4).**
- **ch.03 · Tool system.** Provides the **tool registry with per-tool argument JSON-Schemas**, which §4.5 compiles to the union tool-call grammar so args are **decode-enforced** (`schema → mask_logits`). A grammar dead-end (F4) surfaces back to ch.03 as a tool-arg error. **ch.03 binds `GrammarSpec::ToolCall` / `ToolSchema` (A.2).**
- **ch.04 · Context & memory.** This chapter **provides the `ModelDescriptor`** ch.04 binds (id, arch, ctx_len, tokenizer_sig, footprint — §4.3) and **owns engine selection (transformer vs SSM routing, §4.4), the RoPE/YaRN scaling implementation, KV-precision codecs, and the `embed()`/prefill/copy-KV seams** ch.04 drives. The **`KvStore` interface (ch.04 A.4)** is the runtime-side seam our roles bind; the embedder role powers ch.04's relevance scoring. *Hawking Condense* governs the `.tq` footprint that sets which model fills which role/profile.
- **ch.05 · Codebase intelligence.** Consumes the **embedder** and **reranker** roles (§4.2) — ch.05 already assumes a `/v1/embeddings?role=…` embedder and a listwise/cross-encoder reranker; this chapter is where those roles are *defined and routed*. ch.05's local-LLM listwise rerank == the reranker role on the hero.

---

## Appendix A — Binding contracts

> These are the schemas other chapters import. **Versioned**; additive = minor, breaking = major; consumers pin a major. The two contracts the prompt calls out explicitly — **the model-role + routing contract** and **the grammar/sampler/logit interfaces** — are A.1–A.5.

### A.1 `RoleDescriptor` + `RoleRegistry` (schema_version 1) — the model-role contract

The normative shape is §4.3. The contract other chapters bind:

```rust
struct RoleDescriptor {
    role_kind: RoleKind,        // Hero|FastDraft|Embedder|Reranker|Compactor|Classifier|SsmLong|Custom
    model: ModelDescriptor,     // == ch.04's ModelDescriptor (THIS chapter provides it):
                                //   { id, arch: "transformer"|"ssm", ctx_len_native,
                                //     tokenizer_sig, footprint_mb, quant, embed_capable }
    endpoint: Endpoint,         // resolved by ch.01's router process
    default_sampler: Option<String>,   // a SamplerProfile name (A.3)
    caps: ProviderCaps,         // == ch.01's ProviderCaps + { grammar, logprobs, speculative,
                                //   draft_for: [RoleKind], adapters: [String], max_batch }
    cost: CostModel,            // { ms_per_tok_est, energy_j_per_tok_est } — router input
    escalates_to: Option<String>,      // cascade target (None = top / leaf)
}
type RoleRegistry = Map<String /*role name*/, RoleDescriptor>;  // + RoutingConfig (default_role, escalation, budget_mode)
```

### A.2 Grammar / constrained-decode service (schema_version 1)

`GrammarService::compile(&GrammarSpec, &VocabIndex) -> Arc<CompiledGrammar>` and the `GrammarMatcher { mask_logits, accept_token, is_complete, is_dead_ended }` executor — §4.5.1. `GrammarSpec ∈ { JsonSchema, Gbnf, Regex, Enum, ToolCall{tools:[ToolSchema]}, EditFormat }`. **ch.03 binds `ToolSchema` → `GrammarSpec::ToolCall`.** Cache key: `(blake3(grammar), vocab_sig)`. The executor IS `json_constrain.rs::mask_logits`/`advance`/`is_done`, generalized to a compiled grammar.

### A.3 `SamplerProfile` (schema_version 1)

The struct in §4.6 is normative. Presets `greedy|edit|balanced|explore` are reserved built-ins; custom profiles live under `.hide/profiles/`. Supersets the in-tree `SamplingParams` additively (`min_p`, `typical_p`, `dry`, `logit_bias` are the new fields; the rest are byte-compatible with `engine::SamplingParams`).

### A.4 Logit-level signal shapes (schema_version 1)

```jsonc
// Returned alongside generation when InferenceRequest.want_logprobs (gated, §4.7):
{
  "tokens": [ { "id": 1234, "text": "parse", "logprob": -0.12,
                "top": [ {"id": 1234, "logprob": -0.12}, {"id": 5678, "logprob": -2.4} ] } ],
  "summary": { "mean_token_confidence": 0.91, "first_token_entropy": 1.83,
               "min_token_logprob": -7.2, "self_certainty": 0.74,
               "low_logprob_spans": [ {"start": 14, "end": 17, "min_logprob": -7.2} ] }
}
// Self-consistency (shell-computable today via N calls):
{ "votes": [ {"answer_hash": "blake3:…", "count": 2}, {"answer_hash": "blake3:…", "count": 1} ],
  "agreement": 0.67, "winner": "blake3:…" }
```

### A.5 `AdapterSelection` (schema_version 1)

`{ base_role, adapters: [{id, scale}] }` — §4.9.1. Validated against `RoleDescriptor.caps.adapters`. Recorded in the manifest for replay.

### A.6 Personalization dataset contract (schema_version 1) — the flywheel's data shape

```jsonc
{
  "schema_version": 1,
  "kind": "accepted_edit | rejected_suggestion | tool_call_trace | correction | style_exemplar",
  "input": { "context_manifest_ref": "turn_…", "prompt": "…or messages…" },   // what the model saw
  "output": { "text": "…the diff/tool-call/completion…", "format": "edit|tool_call|text" },
  "label": { "polarity": "positive|negative", "weight": 1.0 },                 // accepted=+ / rejected=−
  "provenance": { "session": "sess_…", "turn": "turn_…", "file": "auth.rs",
                  "accepted_at": "…", "user_modified": false },
  "secrets_scrubbed": true
}
```
Consumed by *Hawking Condense*'s trainer to produce the **personal adapter** (§4.9) / a re-condensed checkpoint. User-inspectable, versioned, deletable.

---

## Appendix B — Prioritized runtime-hooks ask-list

The model layer ships v1 against the **current localhost OpenAI-compatible surface** (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/hawking/generate`, `/v1/hawking/tokens`, `/metrics` + the `json_mode` flag). These are the deeper hooks HIDE wants, **ranked by leverage** (impact ÷ build cost). Each lists the shell-today fallback that holds until it lands. *None is shell-gating.*

| # | Hook | What | Why (leverage) | In-tree starting point | [SHELL-TODAY] until then |
|---|---|---|---|---|---|
| **1** | **Logprobs / confidence in the API** | A `want_logprobs` (+ `top_logprobs: k`) request field; response carries per-token logprob + a confidence/entropy/self-certainty summary (A.4). **Gated readback** (off by default; full/top-k only when asked). | **Unlocks the entire "logits as an API" pillar**: escalation (§4.4), self-certainty best-of-N + early-stop (§4.7), hallucination-span detection. The single highest-leverage decoder hook. | `GenStats` already tracks `readback_bytes`/`logits_materialized_rows`/`token_only_path_used`; the Metal sampler keeps logits on-GPU. The work is *exposing* the values on a gated path. | Self-consistency *voting* via N HTTP calls (vote shell-side) — no hook needed. Per-token confidence waits. |
| **2** | **General grammar / structured-output param** | `response_format: {type: "json_schema", schema}` and/or a `grammar` (GBNF) field and/or strict `tools`; selects a `CompiledGrammar` that drives the mask (A.2), generalizing `json_constrain` from hard-coded JSON to schema/grammar. | **The "small models become reliable" lever** — decode-enforced tool-calls/plans/diffs (§4.5). Frontier-grade format reliability locally. | `json_constrain.rs::mask_logits` is the primitive; `json_mode` already plumbs through `chat_completions`. The work is the compiler front-end + request plumbing + XGrammar-style context-independent precompute. | `json_mode` for generic JSON now; richer schemas via validate-and-retry shell-side. |
| **3** | **LoRA / adapter selection** | An `adapter: [{id, scale}]` request field (A.5) + a runtime adapter loader, low-rank delta in the Metal forward, and a resident-pool manager (S-LoRA pattern). | Per-language + **personal** adapters → much better code in the user's style (§4.9); the flywheel's deploy target (§4.10). | **None in-tree** (clean build). | Serve a merged base+adapter model as its own *role*/endpoint (costs full RAM; only for the 1–2 hottest, incl. a periodically-merged personal hero). |
| **4** | **Sampler superset (per request)** | Additive fields on `SamplingParams`/the request: `min_p`, `typical_p`, `dry`, `logit_bias`, and honoring `seed` everywhere for determinism (A.3). | DRY (kills code-loop repetition), determinism for edits, logit-bias (ban `eval(`, force formats) — quality-of-life + safety, decoder-only (§4.6). | `sample.rs` loop is already mask→penalize→truncate→draw; `SamplingParams` has temp/top_k/top_p/rep_pen/seed. Additive. | `greedy`/`balanced`/`edit`(minus DRY/min-p) work now via temp/top_p/seed. |
| **5** | **KV handles / prefix control** (shared with ch.04) | The `KvStore` ops (ch.04 A.4): `lookup_prefix`, `warm_into_slot`, `demote`, `checkpoint`/`restore`, `stats` — exposed so the orchestrator can pin/reuse/resume per role. | Cross-turn prefix reuse, instant resume, per-role KV residency (ch.04). | `engine.rs` has `prefill_slot_from_pos`, `copy_kv_prefix_to_slot`, `kv_fingerprint_at_pos`; `SystemPromptKvBank` + disk tier shipped. | The shipped bank + disk prefix cache already give cross-request reuse; explicit handles are the upgrade. |
| **6** | **Request-time RoPE/YaRN + arch routing metadata** (shared with ch.04) | A `rope_scaling` override threaded through prefill/forward; `model_arch()` reliably reporting `transformer`/`ssm` for the router. | Per-task context-length dialing (ch.04 §4.4); correct SSM-vs-transformer routing (§4.4). | `model_arch()` exists (defaults `"unknown"` — needs each model to report); RoPE is applied in-forward. | Pick the right *model build*/endpoint per profile; route by registry `arch` field. |
| **7** | **Trained spec head + self-spec** | Load a trained Eagle5/EAGLE-3 head (real accept rate); a layer-skip/early-exit self-spec path for low-RAM. | 3–6× spec on code (§4.8); zero-extra-RAM spec on battery. | `SpeculateMode::Eagle5` + the full proposer fleet/router/governor/bandit shipped; mock head exists. | ExactShared / n-gram / suffix proposers work now via the in-tree router. |
| **8** | **ANE routing + per-role energy accounting** | Route embedder/classifier to ANE/efficiency cores; expose finer per-role J/tok. | Energy/thermal scheduling (§4.11) — quieter, longer battery. | Per-domain J/tok measurement exists; `/metrics` carries some. | Shell reads OS battery/thermal and picks roles/concurrency now. |

---

## Appendix C — Source register

Numbers carrying *(paper-claim)* are authors' reported figures (not independently reproduced) and are gated before entering the bible as measured fact. In-tree facts (the proposer fleet, the verify-cost curve, the energy verdict, `json_constrain`'s masking, RWKV-7 tps) are HIDE's own ground truth.

**Speculative decoding.** Speculative decoding — Leviathan et al., ICML 2023, arXiv:2211.17192; Chen et al., 2023, arXiv:2302.01318. · Medusa — Cai et al., ICML 2024, arXiv:2401.10774. · EAGLE / EAGLE-2 — Li et al., 2024, arXiv:2401.15077 / arXiv:2406.16858. · **EAGLE-3** — Li et al., 2025, arXiv:2503.01840 (direct-token draft + training-time-test; 3–6.5×; accept length ~4.5–5.0, flat across positions; ~532K train examples). · LayerSkip / self-speculative — Elhoushi et al., ACL 2024, arXiv:2404.16710; Zhang et al., 2023, arXiv:2309.08168. · Lookahead decoding — Fu et al., ICML 2024, arXiv:2402.02057. · REST (retrieval drafting) — He et al., NAACL 2024, arXiv:2311.08252. · Scaling laws for spec-decode — 2025, arXiv:2505.07858. · MARS (margin-aware verify) — 2026, arXiv:2601.15498.

**Constrained / structured generation.** Outlines (regex/JSON→FSM) — Willard & Louf, 2023, arXiv:2307.09702. · **XGrammar** — Dong et al., 2024, arXiv:2411.15100 (context-independent/dependent split + pushdown automaton; up to 100×; near-zero serving overhead). · XGrammar-2 — MLC, 2026, arXiv:2601.04426 (up to 80× over XGrammar; cross-grammar caching; JIT). · Pre³ / DPDA — 2025, arXiv:2506.03887. · GBNF — llama.cpp grammars (de-facto local format). · JSONSchemaBench — 2025, arXiv:2501.10868 (empirical schema-validity + task-quality effects). · "Thinking Before Constraining" — 2026, arXiv:2601.07525 (over-constraining hurts reasoning; think→emit). · LMQL — Beurer-Kellner et al., 2023.

**Model routing.** RouteLLM — Ong et al., ICLR 2025, arXiv:2406.18665 (matrix-factorization router: 95% GPT-4 quality at 26%→14% GPT-4 calls; ~48%→75% cheaper). · LLMRouterBench — 2026, arXiv:2601.07206. · IPR (user-controlled quality-cost) — 2025, arXiv:2509.06274.

**Samplers.** min-p — Nguyen et al., ICLR 2025, arXiv:2407.01082; critique — 2026, arXiv:2506.13681 (human-eval null on quality/diversity Pareto). · Locally-typical — Meister et al., TACL 2023, arXiv:2202.00666. · mirostat — Basu et al., ICLR 2021, arXiv:2007.14966. · DRY — community-proven sequence-aware repetition penalty (oobabooga/llama.cpp). · p-less sampling — 2025, arXiv:2509.23234.

**Logit confidence / test-time compute.** Self-consistency — Wang et al., ICLR 2023, arXiv:2203.11171. · Self-certainty / Best-of-N — 2025, arXiv:2502.18581. · Deep Think with Confidence — 2025, arXiv:2508.15260. · Self-Calibration (one-pass confidence) — 2025, arXiv:2503.00031. · Prefix-confidence at test time — 2025, arXiv:2507.18122. · Multi-Agent Verification — 2025, arXiv:2502.20379. · Value-back RL test-time scaling — 2025, arXiv:2505.04842. · Certified self-consistency — 2025, arXiv:2510.17472.

**LoRA serving.** S-LoRA — Sheng et al., MLSys 2024, arXiv:2311.03285 (Unified Paging + heterogeneous batching; thousands of adapters/GPU; up to 4× throughput vs PEFT/vLLM). · Punica/SGMV — 2023 (multi-LoRA batching). · Heterogeneous LoRA in distributed serving — 2025, arXiv:2511.22880.

**Activation steering / RepE.** ActAdd — Turner et al., 2023, arXiv:2308.10248. · CAA — Rimsky et al., ACL 2024, arXiv:2312.06681. · RepE — Zou et al., 2023, arXiv:2310.01405. · Steering-vector reliability/geometry limits — 2026, arXiv:2602.17881.

**In-tree (HIDE's own ground truth).** `crates/hawking-core/src/engine.rs` (`Engine` trait, `SamplingParams`, `GenerateRequest.json_mode`, `GenStats::dec_tps`/`draft_accept_rate`, prefill/copy-KV/embed seams). · `crates/hawking-core/src/sample.rs` (CPU+Metal sampler; logits-on-GPU, 4-byte argmax readback). · `crates/hawking-core/src/json_constrain.rs` (`JsonConstraint` + `JsonVocabIndex` + `mask_logits` — the grammar primitive). · `crates/hawking-core/src/speculate/` (`ProposerId` ×8, `router.rs` `verify_cost_forwards` curve B=8≈4.15, EWMA cost models; `governor.rs`/`spec_gov.rs` accept-rate auto-disable; `policy.rs` UCB1 bandit; `eagle5*` mock head). · `crates/hawking-serve/src/http.rs` (`/v1/*` routes, `ResponseFormat{json_object}→json_mode`). · `crates/hawking-core/src/tq.rs` (`.tq`=STR2; CPU parity oracle for the staged GPU bitslice GEMV; *Hawking Condense* binding). · Models: `qwen_dense.rs` (hero), `rwkv7.rs`/`mamba2.rs` (SSM, O(1) state ~6 MiB), `llama.rs`/`deepseek_v2.rs`/`gemma2.rs`/`phi3.rs`/`mixtral.rs`/`qwen_moe.rs`/`olmoe.rs`. · RWKV-7 measured flat ~118→119 tps to 8k vs Qwen ~40→8.6 (in-tree campaign). · Per-domain energy J/tok measurement (the "genuine" energy verdict).
