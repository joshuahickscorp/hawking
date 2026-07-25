# Hawking Ascension — CLOSED

```
endpoint: HAWKING_ASCENSION_CLOSED
sealed:   2026-07-24
branch:   campaign/glm52-generation-b
```

## The one question, answered

> Does the functional student compose across the full model?

**No.** Independent per-layer functional students do not compose — on GLM (an expansive
residual stream) or on DeepSeek (router divergence under cumulative drift) — even though the
functional escape exists and is locally strong on both.

## DeepSeek final verdict: FUNCTIONAL_PARTIAL_ONLY

The full cascade put a functional student in all 40 sparse MoE layers of the 43-layer model,
on the validated streamed forward.

```
first block-skill divergence   layer 8
router top-1 below half         layer 4
final block skill               -168.6
mean per-layer amplification    1.121
router top-1 (min)              0.0
greedy token agreement          0.0
L3 first student (pre-drift)    skill 0.931, router top-1 1.0
```

The mechanism is decisive and robust to student quality: the first student at layer 3 fits
well, skill 0.93 with perfect routing, but its small error drifts the hidden state enough
that the **very next router** (layer 4) picks different experts. Once routing diverges the
wrong experts run, and mild per-layer amplification (1.12 mean) compounds over forty layers
(1.12⁴⁰ ≈ 74×) into divergence. Any nonzero per-layer error — and no independent student is
error-free — accumulates into route divergence. The composition failure is structural.

This reconciles with the favourable single-layer probes (L5 contractive, L20 near-neutral,
L38 mild threshold): a single injected perturbation is absorbed, but a per-layer error chain
is not, because the router re-selects experts once drift accumulates. **Single-layer
amplification favourability did not imply composition, and the cascade is the decisive
test.**

## Required closure fields

| field | result |
|---|---|
| DeepSeek verdict | `FUNCTIONAL_PARTIAL_ONLY` (cascade refuted) |
| cascade depth / first divergence | 40 sparse layers / block-skill L8, router L4 |
| complete-model BPW | student_h1024 0.41 source-protected, 0.34 int8, 0.18 int4 |
| runtime path working | streamed forward embedding→43 layers→logits→token demonstrated; GLM `.gravity`→block→token proven |
| true TPS measured | GLM functional MoE-path 6.3–7.1 TPS (measured Metal); DeepSeek roofline 35.7 teacher / 54 functional (measured BW / active bytes); wall-clock served-TPS UNAVAILABLE (no resident compact model) |
| FLOP/byte/energy | GLM FEWER_FLOPS_AND_BYTES; DeepSeek FEWER_BYTES_MODEST (attention-bound); energy UNAVAILABLE, none inferred |
| HIDE contracts frozen | `HAWKING_HIDE_HANDOFF_CONTRACT.json` — full surface |
| commits pushed | yes |
| system clean | yes |

## Storage

Both parents released under the one-parent policy, each re-fetchable at its immutable
revision against a sha256 rehydration receipt:

```
GLM-5.2         RELEASED  405.4 GB reclaimed  76/76 shard sha256
DeepSeek-V4     RELEASED  159.6 GB reclaimed  46/46 shard sha256  (terminal; meta + evidence + validated forward retained)
free                      539 GiB
```

## What ascended, scoped

`HAWKING_CORE_ASCENDED` — methodology, metric system, `.gravity` substrate, source
lifecycle, Apple runtime foundation, functional-runtime discovery, HIDE-facing architecture.

The cross-parent functional paradigm is **CLOSED**: independent per-layer functional
students do not compose. The metric is not broken. There is no winning servable
architecture, and none is claimed.

## Durable results

- the null-first, amplification-first, **cascade-decisive** methodology
- the validated streamed-forward apparatus: validate primitives against the official
  reference, stream one block at a time, use contextual inputs
- **single-layer favourability does not imply composition** — the cascade is the test
- the functional `.gravity` codec, CPU authority, and parity-gated Metal grammars
- GLM and DeepSeek both `FUNCTIONAL_PARTIAL_ONLY`, by different mechanisms

## Next phase

Refactor, release selection, and HIDE completion. **Do not reopen the functional search
space.** A route-aware, jointly-trained student is the only identified path to composition
and is a separate future research decision, not part of this closure.

> Gravity is the preservation of a stable causal trajectory under the smallest executable
> physical system. The independent-per-layer student is not that system, because the
> trajectory it must preserve runs through a router that will not tolerate its drift.
