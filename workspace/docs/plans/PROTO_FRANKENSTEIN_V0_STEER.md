# PROTO_FRANKENSTEIN_V0 — FULL LATENT GLM→DSV4F TRANSFER (owner steer, canonical)

Hard deadline: **Friday 2026-08-07**. Fully **local / free** (NO paid API).

## Definition (frozen)
PROTO_FRANKENSTEIN_V0 = a real, trained, **full latent** GLM-5.2 → DeepSeek-V4-Flash
transfer. Behavior-only distillation is supplementary. The closed-form linear map is
`LINEAR_SUBSPACE_INITIALIZATION` (cartography + bridge init + negative control) only.
Neither qualifies as V0 alone. NOT Kimi, NOT Odyssey, NOT Gravity recomposition.
The deadline concession is **corpus size, not transfer completeness.**

## The key feasibility unlock — teacher-forced, layer-major GLM (NOT a chat server)
Full latent transfer does NOT need an interactive GLM decode server. It needs an
**exact teacher-forced forward** over a frozen batch of sequences, executed layer-major:
```
freeze a batch of sequences
→ stream GLM layer L once → run ALL sequences/microbatches through it
→ capture bounded hidden-state outputs → atomically seal next-layer states
→ evict layer-L weights → stream layer L+1 → … through all 78 layers
```
Amortizes each streamed GLM layer across the whole corpus; never holds GLM+DSV4F
resident. Double-buffer prefetch (N-1 seal/evict, N execute, N+1 download/verify).
Evict on failure. Preserve the 25 GiB floor + the source-only reclaim allowlist.

## Bounded full-latent corpus (ladder — scale corpus, never cut the mechanism)
Disjoint memberships: TRAIN / CALIBRATION / PUBLIC_TEST / HIDDEN_TEST / RETENTION.
L0: 32 seqs (pipeline smoke, all 78 layers) → L1: 128 (first full train) →
L2: 256–512 (primary V0) → L3: larger only if held-out gain + time. Real verified
problems (math method selection, multi-step, formalization, proof repair,
counterexample, symbolic, coding, repo, tools, agent, long-ctx, general). NO synthetic
Gaussian activations.

## Tokenizer-independent alignment (never token-ID↔token-ID)
Per shared text: UTF-8 byte spans, GLM-token→byte-span, DSV4F-token→byte-span, shared
semantic anchors (claim boundaries, proof steps, subgoals, code AST regions, tool/
formal actions, answer spans). Pool activations over corresponding byte/semantic spans.

## Capture both models (bounded; samples + sufficient statistics + small trace shards)
GLM: embedding, early pre/post attn, early pre/post MoE, mid pre-router, mid router
logits+experts, mid post-MoE, late proof/answer states, final norm, final logits,
method/decomposition/repair labels where available.
DSV4F: embedding, mHC, pre/post attn, pre-router, router logits/top-6/routes/margins,
post-MoE, late hidden, logits, HCLI action/tool decisions.

## Layer/phase correspondence (measure, don't ratio 78→43)
GLM×DSV4F CKA, CCA, Procrustes residual, functional-intervention sensitivity, causal
tracing where affordable. Monotonic many-to-one phase alignment (lexical/context,
early reasoning, method selection, planning/decomposition, tool/formal prep, repair/
critique, answer/proof consolidation). Seal GLM_DSV4F_LAYER_CORRESPONDENCE.json +
GLM_DSV4F_PHASE_ALIGNMENT.json.

## Full latent bridge architecture (shared latent space, reversible student adapters)
Teacher projector: GLM 6144 → RMSNorm → learned proj → shared latent.
Student observer: DSV4F 4096 → RMSNorm → learned proj → same shared latent.
Student intervention: DSV4F hidden → RMSNorm → low-rank proj → gated nonlinear MLP →
low-rank residual → add back to native hidden. Init from LINEAR_SUBSPACE_INITIALIZATION
where useful. Multiple bridge sites (early context, mid method/planning, pre-router,
post-MoE, late consolidation, final/value). Runtime keeps ONLY the DSV4F adapters/
projectors/heads; GLM teacher projectors + raw activations are training-only.

## Required V0 modules (reversible, bypassable, hash-bound, ablatable, Gravity-accounted)
GLM_EARLY_CONTEXT_BRIDGE, GLM_METHOD_BRIDGE, GLM_DECOMPOSITION_BRIDGE,
GLM_PRE_ROUTER_BRIDGE, GLM_POST_MOE_BRIDGE, GLM_FORMALIZATION_BRIDGE, GLM_REPAIR_BRIDGE,
GLM_LATE_CONSOLIDATION_BRIDGE, GLM_VALUE_HEAD, GLM_METHOD_CONDITIONED_ROUTE_RESIDUAL.
No direct GLM router/expert weight transplant. Must stay Kimi-bridge compatible.

## Router distillation (semantic policy, not IDs)
Method/action classes (algebra, geometry, combinatorics, formal proof, symbolic,
counterexample, retrieval, coding, tool, verification, repair). Bounded route residual
that preserves native top-6 semantics, load balance, route margins, route-set
stability, non-math routing, runtime predictability. Reject route collapse.

## Loss portfolio (never latent-cosine alone)
L_latent (shared-space feature align), L_function (functional checkpoint agreement),
L_span (aligned decoded-span distribution), L_method, L_decomposition, L_formal,
L_repair, L_value (verified outcome/value rank), L_route (semantic route policy),
L_retention (base-capability preservation), L_runtime (adapter sparsity/cost). High
CKA that fails held-out behavior is rejected.

## Retention / anti-catastrophic (freeze BASE_DSV4F on RETENTION split)
Preserve coding, repo, tools, agent, general, conversation, long-ctx, HCLI, structured
JSON, runtime. base-logit KL + base feature anchoring + mixed batches + adapter gating
+ small reversible updates. Reject math gained via material secondary regression.

## Ablation matrix (complete must beat linear init)
A BASE_DSV4F, B LINEAR_SUBSPACE_INITIALIZATION only, C LATENT_BRIDGES only,
D BEHAVIOR_HEADS only, E LATENT+BEHAVIOR, F LATENT+ROUTE_RESIDUAL, G COMPLETE V0.
Report per arm: held-out math, method/decomp/formal/repair/counterexample, coding,
tools, agents, general, route stability, artifact bytes, resident bytes, active
bytes/token, TPS/p99.

## Training schedule + checkpoints
A projectors+shared-latent align → B reversible student latent adapters → C method/
decomp/formal/repair/value heads → D bounded route residual → E joint consolidation
w/ retention replay → F hidden eval + repair. Keep CURRENT / BEST_MATH / BEST_BALANCED
/ ROLLBACK. Evict superseded checkpoints ONLY after verified cloud upload.

## Promotion gates (all required)
Real GLM activations from the complete 78-layer teacher-forced path used; nonlinear
latent bridges trained; complete intended bridge classes exist; held-out math improves
over BASE; complete V0 beats linear init; method/decomp/repair improve; no material
secondary regression; routing stable; reversible+loadable; runtime/storage sealed;
KIMI_STRATEGIC_BRIDGE intact; exact provenance; independent verification. Reject gains
from contamination or teacher-answer memorization.

## Failure policy
Fix ordinary failures autonomously. If a bridge architecture fails: retain evidence,
try a materially different nonlinear bridge / different correspondence sites / different
shared-latent width / different gating / different loss balance / rerun bounded
training. Do NOT return after the first failed bridge. Do NOT declare it impossible
from the linear mapper. Reduce corpus before removing latent-transfer completeness.

## Terminal endpoint
`PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED` — complete 78-layer streamed GLM teacher-
forced activation path + paired DSV4F path + tokenizer-independent alignment + measured
correspondence + trained nonlinear latent bridges + method/decomp/formal/repair/value
modules + semantic route residual + retention/non-regression gates + full ablation +
loadable reversible DSV4F artifact + Kimi bridge preserved + provenance + storage/
runtime receipts + verified cloud upload + one-command restore + raw GLM windows and
superseded checkpoints evicted. Do NOT return scaffolding/mapping/behavior-only/pending.
