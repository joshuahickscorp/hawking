# Architecture

EAGLE-4 is a single transformer block plus three output heads. Everything
flows from one design choice: **the head starts as the target model's own
LM head, and training only learns the small refinement.**

## Core trick: residual gate

```
draft_hidden = post_norm(h_high) + α · block(in_proj(prev_embed, h_low, h_mid, h_high, h_shared))
```

`α` is a learned scalar, initialized to `0.05`. The block path is small at
init but non-zero — at step 1 it nudges `draft_hidden` slightly off
V2-Lite's own `post_norm(h_high)`. Training is a search for an `α` and a
`block(·)` that align the head's argmax with the target's.

(Earlier versions inited `α = 0` and relied on the corpus-CE gradient to
move it. That worked but bootstrapped slowly, because once the loss
ramps to target-argmax CE the gradient is near-zero at exact identity.
A tiny non-zero α at init keeps gradient flowing through the block from
step 1.)

## Inputs (per token)

Five vectors captured from one V2-Lite forward pass:

| input | shape | source |
|---|---|---|
| `prev_token` | int32 | the token whose successor we're predicting |
| `h_low` | fp16[2048] | output of decoder layer 2 |
| `h_mid` | fp16[2048] | output of decoder layer 13 |
| `h_high` | fp16[2048] | output of decoder layer 25 (last before final norm) |
| `h_shared` | fp16[2048] | last MoE layer's shared-expert output |

`capture.py`'s monkey-patched MLP hook captures all five from one forward
pass. ~745 records/sec on M3 Pro.

## Forward

```
prev_embed     = V2Lite_token_embd[prev_token]           # frozen lookup
x              = concat(prev_embed, h_low, h_mid, h_high, h_shared)
x              = Linear(5H → H)(x)                       # learned in_proj
x              = TransformerBlock(x, diagonal_attn_mask) # learned, ~50M params
draft_hidden   = post_norm(h_high) + α · x               # residual gate
token_logits   = draft_hidden @ V2Lite_lm_head           # frozen lm_head
mask_logits    = Linear(H → 26·64)(SiLU(Linear(H → 512)(draft_hidden)))
calib_logit    = Linear(H → 1)(draft_hidden)             # P(accept) at sigmoid
```

The attention mask is diagonal (each position is independent) — every
training row is its own example, packed (B, S) for throughput, not because
positions interact.

## Losses

Four terms, weighted:

```
L = α_step · CE(token_logits, V2Lite_argmax)    # hybrid: ramps in over 500 steps
  + (1 − α_step) · CE(token_logits, next_token) # corpus CE for warm start
  + 0.5  · MSE(draft_hidden, post_norm(h_high)) # identity pull
  + 0.3  · BCE(mask_logits, routed_mask)        # routing
  + 0.1  · BCE(calib_logit, head_argmax == V2Lite_argmax)  # calibration
```

`α_step` linearly ramps from 0 to 1 over the first 500 training steps,
then stays at 1. Aligning the optimized objective with the eval metric
(`head.argmax == V2Lite.argmax`) is what flips single-step target-argmax
acceptance from 87% → 95% over the v2 trainer.

First 3 positions of each sequence are excluded — BOS-adjacent tokens
have ~17% smaller hidden L2 norm and dominate early training otherwise.

## Eval — τ-at-depth-K

Single-step target-argmax is a weak metric: at init the head is exactly
identity-on-h_high and trivially scores 100%. The real spec-decode metric
is **τ-at-depth-K**: roll the head out autoregressively for K steps,
feeding its own argmax as the next `prev_token`, and count how many
tokens are accepted before the head disagrees with V2-Lite at any depth.

We report τ at depth 4 against the EAGLE-3 baseline on the same heldout
shard. v3 hits τ=3.57, EAGLE-3 hits τ=2.15.

## Why MoE routing matters

Standard speculative decoding (EAGLE-3, Medusa, etc.) accepts or rejects
the draft's token. On MoE targets, even a *rejected* draft has done
useful work if it correctly predicted which experts the target will need:
the runtime can prefetch those weights before verify hits them, turning
a memory-bandwidth-bound verify into a compute-bound one.

The 26 × 64 mask head is the artifact that enables that. Top-8 recall is
17–21% (v3-spec vs v3-routing trade) against a ~9% random baseline.

## What's pending

- **Wall-clock tps.** τ=3.57 is offline acceptance; turning it into
  tokens-per-second requires the spec-decode runtime — that's the
  dismantle integration, separate repo.
- **Stronger mask head.** Top-8 recall plateaus around 21% in this
  trainer. Pushing it harder is the obvious next run.

## Why this is structurally simple

Everything derives from one trick (residual gate) and one data format
(five captured vectors per token). Two Python files contain the entire
training pipeline; two more do bench and τ eval:

- `eagle4.py` — head class, training, eval, quantize, frozen-weight extract
- `capture.py` — V2-Lite forward with per-layer routing hooks → parquet
- `bench.py` — EAGLE-3 baseline trainer + compare
- `tau_eval.py` — τ-at-depth-K eval

The runtime is structurally separate and lives in dismantle.
