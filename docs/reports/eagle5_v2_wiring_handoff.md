# Eagle5 v2 (activation-sparsity head) — wiring handoff

**Status:** design doc + scaffold code landed. Trainer + τ-eval shipped as
runnable Python under `tools/training/`. Training itself is the user's
overnight run (5–10 h on M3 Pro); this doc spec'd what to expect and how
to gate.

**Projected gain:** **+5 to +12 dec_tps** *IF* the trained head meets
τ-at-depth-4 ≥ 3.0 AND a spec-decode runtime that converts τ to
wall-clock tps lands. Both gates are non-trivial — see [[path-to-100-repath]]
for why current spec-decode runtimes regress vs off-mode.

**This doc covers the head only.** The runtime wiring (Rust + Metal
spec-decode that consumes the head's outputs) is a SEPARATE workstream.
The deliverable here is a trained + quantized head and a τ-at-depth-K
score against the heldout corpus shard.

---

## 1. Why eagle4 isn't enough already

Eagle4 already ships at τ-at-depth-4 = 3.57 on V2-Lite-Chat. Why eagle5?

- **Eagle4's routing-mask head doesn't help speed yet.** The 26×64 routing
  mask predicts which experts each layer routes; eagle4's top-8 recall is
  17–21%. That signal could drive expert-prefetch, but the spec-decode
  runtime that would consume it doesn't exist in dismantle (per
  [[path-to-100-repath]]: "all spec-decode REGRESSES").
- **The calibration analysis (2026-05-21) killed the routing-prediction
  angle entirely.** Per-layer balance scores are 0.987–0.995 across all
  26 MoE layers — routing is essentially uniform. A perfect mask
  predictor wouldn't help, because there's no concentration to exploit.
- **A different signal CAN help.** Per-token activation sparsity in the
  expert FFN intermediate channels. Many tokens activate only a small
  subset of intermediate channels strongly; predicting those channels
  enables sparse decode (skip dense GEMM ops over inactive channels) OR
  a smaller draft model that mimics the full model's token argmax.

Eagle5 v2 builds the second kind of head: predicts (a) the next token
*and* (b) which intermediate channels carry signal for the next token.
Same training pipeline shape as eagle4; new architectural inputs and a
new auxiliary head.

## 2. What you have on disk

### 2.1 Corpus (training inputs)

- `artifacts/calibration/v2_lite_corpus/shard_*.parquet` (141 shards,
  4,512 sequences × 256 tokens)
- Columns relevant here:
  - `tokens`: input ids (n_tok int32)
  - `residual_in_per_layer`: (27 layers × n_tok × 2048 fp16 → quant int8
    per the manifest's `quantize_intermediates: "int8"`)
  - `intermediate_per_layer`: **per-MoE-layer first-expert FFN output**,
    captured via `mlp.experts[0].register_forward_hook` in
    `tools/training/build_corpus.py:202`. Shape: (26 layers × n_tok ×
    2048 hidden). NOT the SwiGLU intermediate (1408 channels) — it's the
    post-down hidden-space output of the FIRST expert (id 0) per layer.
    **This is the key caveat — see §5.4.**
  - `expert_idx_per_layer`: (26 × n_tok × top_k=6) — useful for the
    auxiliary routing head if you want to keep it.
  - `routing_topk_weight_per_layer`: (26 × n_tok × top_k) — gates.

### 2.2 Eagle4 reference

- `eagle4/eagle4.py` — head + train + eval + quantize (~500 LOC, MLX)
- `eagle4/tau_eval.py` — τ-at-depth-K metric
- `eagle4/q4_parity.py` — q4 quantize parity check
- `eagle4/v2lite_frozen.npz` — frozen V2-Lite weights for the head's
  built-in lm_head + token_embd + output_norm references

### 2.3 New artifacts in this session

- `tools/training/eagle5_train.py` — trainer; mirrors eagle4/eagle4.py
  shape; new inputs and losses per §3 below
- `tools/training/eagle5_tau_eval.py` — τ + channel-sparsity recall eval
- `tools/training/eagle5_quantize.py` — q4 head export + parity

## 3. Architecture

```
prev_embed     = V2Lite_token_embd[prev_token]
residual_in    = residual_in_per_layer[capture_layer]      # (B,S,2048)
inter_signal   = intermediate_per_layer[capture_layer]     # (B,S,2048)
x              = concat(prev_embed, residual_in, inter_signal)        # (B,S, 3*2048)
x              = Linear(3H → H)(x)                                    # learned in_proj
x              = TransformerBlock(x, diagonal_attn_mask)              # 1 block, ~25M params
draft_hidden   = post_norm(residual_in) + α · x                       # residual gate (init α=0.05)
token_logits   = draft_hidden @ V2Lite_lm_head                        # frozen LM head
sparsity_log   = Linear(H → 1408)(SiLU(Linear(H → 512)(draft_hidden)))  # per-channel logits over MoE FFN intermediate
calib_logit    = Linear(H → 1)(draft_hidden)                          # sigmoid → P(accept) like eagle4
```

**Capture layer:** layer 25 (the last layer before final norm). Same
as eagle4's `h_high`. Eagle5 v2 uses ONE layer's residual + intermediate
instead of eagle4's three (low/mid/high) — simpler head, fewer
parameters. If τ doesn't hit 3.0, scale back to eagle4's 3-layer mix.

**Param budget:** ~25–30M trainable. In_proj 3*2048×2048 ≈ 12M;
TransformerBlock with 16 heads ≈ 12M; sparsity head 2048*512+512*1408 ≈
1.8M; calib ≈ 2K. Total ~26M (matches eagle4's ~30M).

**Why MLX:** matches eagle4. Existing `eagle4/v2lite_frozen.npz`
loader can be reused with minor cleanup. PyTorch isn't measurably
better given the Apple-silicon-only context.

## 4. Loss

Three terms, weighted to match eagle4's empirical balance:

```
L = α_step  · CE(token_logits, V2Lite_argmax)              # hybrid ramp
  + (1-α_step) · CE(token_logits, next_token)              # corpus warm-start
  + 0.3 · BCE_with_logits(sparsity_log, channel_active_gt) # new: sparsity head
  + 0.1 · BCE_with_logits(calib_logit, head_argmax == V2Lite_argmax)
```

`α_step` ramps from 0 to 1 over 500 steps, same as eagle4.

**Channel sparsity ground truth.** `channel_active_gt[b,s,c] = 1 iff
|intermediate_per_layer[capture_layer, b, s, c]| > threshold_p90`. The
threshold is computed per-batch as the 90th percentile across all
channels in the batch — a calibration that the trainer prints at step 0
for sanity. This represents "the top-10% most-active channels for this
token" — a 200ish-channel-out-of-1408 sparse mask.

NOTE: `intermediate_per_layer` was captured from expert 0 only, NOT the
actual mixture. See §5.4 for the limitation discussion.

## 5. Training pipeline

### 5.1 Run

```bash
python3 tools/training/eagle5_train.py \
    --corpus-dir artifacts/calibration/v2_lite_corpus \
    --frozen      eagle4/v2lite_frozen.npz \
    --ckpt-dir    checkpoints/eagle5_v2 \
    --epochs      5 \
    --batch-size  16 \
    --seq-len     16 \
    --lr          3e-4 \
    --capture-layer 25
```

Expected runtime: ~5–10 hours on M3 Pro 18GB.

### 5.2 Checkpoints

`checkpoints/eagle5_v2/latest.npz` + `step_NNNNNN.npz` every 200 steps.
The `step_*` files are kept; the trainer prunes them only if storage
overflows.

### 5.3 Logging

`checkpoints/eagle5_v2/log.jsonl` — one row per 25 steps: loss
components, gate α, learned residual_gate, wall time. Mirrors eagle4.

### 5.4 The expert-0 caveat

`intermediate_per_layer` in our corpus is the output of `mlp.experts[0]`
only. For most tokens, expert 0 is NOT among the top-6 routed experts;
its output is what we'd get IF that expert had been routed but with the
attention/residual stack feeding it as if it were. So the "channel
activity" we're predicting is **expert 0's per-token channel response**,
NOT the per-token aggregated routed-expert response.

Two implications:

1. **The sparsity head's ground truth is noisy.** Channel-active for
   expert 0 may not equal channel-active for the actually-routed top-k
   experts. The predictor is trained against a proxy, not the truth.
2. **Training quality bound.** If expert-0 channel activity correlates
   well with routed-mixture channel activity (which it might, since
   shared experts plus learned-to-cooperate routed experts often have
   similar activation patterns), the predictor still helps. If not,
   the sparsity head is just learned noise.

**Mitigation options (gate during training):**

- **A. Use only the routing-aware MEAN over top-k.** Re-capture corpus
  with all top-6 experts' intermediates — cost: ~3 hours of capture
  compute. Larger storage. Cleaner ground truth.
- **B. Treat expert 0 as a proxy and accept the noise.** Useful if
  expert 0 is representative; the trainer measures sparsity-head BCE
  per-step so we'd notice early if loss never drops.
- **C. Drop the sparsity head entirely.** Keep just the token-prediction
  head (essentially "eagle3 with our inputs") and ship τ-only. Saves
  training time; loses the channel-prediction angle.

The trainer accepts `--sparsity-head [proxy|off]` with default `proxy`.

## 6. Eval — τ-at-depth-K (the metric that translates to wall-clock)

`tools/training/eagle5_tau_eval.py` mirrors `eagle4/tau_eval.py`:

```bash
python3 tools/training/eagle5_tau_eval.py eval \
    --ckpt          checkpoints/eagle5_v2/latest.npz \
    --frozen        eagle4/v2lite_frozen.npz \
    --corpus        artifacts/calibration/v2_lite_corpus \
    --depth         4 \
    --max-windows   2000
```

Reports:
- `tau` — mean accepted prefix length (target: ≥ 3.0; stretch ≥ 3.57 to
  match eagle4)
- `per_pos_accept_rate` — depth-1 / depth-2 / depth-3 / depth-4 accept
- `sparsity_top_k_recall` — top-200 channel recall (target: ≥ 0.40,
  meaningful signal vs random 0.14 baseline)

## 7. Quantization

`tools/training/eagle5_quantize.py` uses MLX's `mx.quantize` (group_size=64,
bits=4) on the head's linear-layer weights — same recipe as eagle4. A
parity check generates `argmax(bf16) vs argmax(q4)` over 1000 corpus
tokens; ship-gate: ≥ 99% match.

## 8. Acceptance gates (lever 3)

| gate | target | rationale |
|---|---|---|
| τ-at-depth-4 | ≥ 3.0 | better than eagle3 (2.15); needed for >+5 tps gain even with a working runtime |
| depth-1 accept | ≥ 85% | first-token-correct rate; eagle3 was 73.8% |
| q4 head parity | ≥ 99% argmax match vs bf16 | matches eagle4 |
| sparsity recall@200 | ≥ 0.40 | optional; only relevant if `--sparsity-head=proxy` |

**Bit-identical greedy is NOT required.** The head is probabilistic; it
DOESN'T verify, the full V2-Lite model verifies. The runtime arbitrates
correctness via the verify step.

## 9. Runtime wiring (out of scope here)

What the Rust+Metal spec-decode runtime needs to consume eagle5 v2:

- Input format: residual stream at capture layer + intermediate stream
  at same layer + prev_token id. All available in the existing forward
  pass at layer 25.
- Output format: `(token_logits: [vocab], sparsity_log: [moe_intermediate])`
  per draft step.
- Channel sparsity top-K: compute on-device or on-host? On-device
  (Metal sort kernel) is cheap; we'd need a new shader. On-host
  (Rust topk over the 1408-channel logit) is ~µs.
- Storage format: safetensors (cleaner than GGUF for Python-trained
  heads).

**The current spec-decode runtime regresses vs off-mode by 33-72% per
`reports/path_to_90/_bench_20260520T143008/`.** Per
[[path-to-100-repath]], recovering it is its own track: locate the
eagle4-K=1 8.9-tps tax, then diagnose chain-K=4 acceptance. Eagle5 v2
landing here doesn't fix that. Both tracks need to converge before
+tps shows up.

## 10. File-level diff summary

| file | status | LOC |
|---|---|---|
| `tools/training/eagle5_train.py` | new | ~350 |
| `tools/training/eagle5_tau_eval.py` | new | ~200 |
| `tools/training/eagle5_quantize.py` | new | ~100 |
| `reports/eagle5_v2_wiring_handoff.md` | new (this doc) | — |
| `checkpoints/eagle5_v2/` | created at first training run | — |

The trainer reuses eagle4's `v2lite_frozen.npz` loader via the
`eagle4` Python package (already pip-installable per `eagle4/pyproject.toml`).

## 11. What NOT to do (lever 3)

- Don't try to predict expert routing. Calibration says 0.987-0.995
  uniform balance — no exploitable concentration.
- Don't reach for 6+ inputs (eagle4 had 4). Three is sufficient for the
  V2-Lite hidden_dim; more inputs just inflate in_proj.
- Don't design the spec-decode runtime here. That's a Rust workstream
  contingent on Step 2A diagnosis in `path-to-100-repath`.
- Don't gate ship on absolute tps. Gate on τ + accept rate; the
  Rust runtime is what translates those to tps.
- Don't quantize lower than q4. Eagle4 hit 99.9% argmax-parity at q4;
  q3 or below loses the calibration signal.

## 12. Realistic timeline

| step | duration | who |
|---|---|---|
| Train v1 (proxy sparsity head) | 5–10 h M3 Pro | overnight user run |
| τ eval + iterate (loss balance, capture layer) | 2-3 days | user with assistance |
| Q4 quantize + parity | 1 hour | user |
| If τ < 3.0: consider corpus re-capture with all-experts intermediate | +3 h capture + retrain | user |
| Rust+Metal runtime wiring | separate workstream | separate session |

## 13. Related memory

- [[corpus-complete-analysis-landed]]
- [[path-to-100-repath]] (spec-decode runtime context)
- [[v110-path30-findings]]
