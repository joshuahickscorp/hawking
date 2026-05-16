# Path-to-90 C2 — Training session brief (EAGLE-3 head for V2-Lite)

**Audience:** the next session that trains the draft head.
**Prereq satisfied by:** this commit ships the data pipeline + a 5K-sample capture run; this brief tells the trainer what to do with it.
**Architecture decision:** [stage3_c1/architecture.md](../stage3_c1/architecture.md) (EAGLE-3, default).
**Compute decision (added 2026-05-16):** local MLX training on M3 Pro, NOT H100 rental. Implementation stack lives at [tools/training/mlx_eagle/](../../../tools/training/mlx_eagle/). See [tools/training/mlx_eagle/README.md](../../../tools/training/mlx_eagle/README.md) for the sequenced step-by-step.

## TL;DR

Train a 1-block transformer (≈ 60 MB, fp16) on captured `(hidden_state, next_token)` pairs. Hidden state is the **post-final-rmsnorm** output of DeepSeek-V2-Lite (h=2048). Next token is the **teacher-forced** next position in the source dialogue. The trained head, plus a verify-side rewrite (Path B, separate work), unlocks the spec-decode regime for path-to-90.

## Inputs

| Path | Description |
|---|---|
| `training_data/c2_hidden/eagle3_v0/shard_*.bin` | DCAP v1 binary shards, ~5K samples / ~600K records as of session 2026-05-15. Re-runnable / extensible via `dismantle capture-hidden --resume`. |
| `training_data/c2_hidden/eagle3_v0/shard_*.meta.json` | Sidecar with hidden_dim=2048, dtype=float16, model_id=DeepSeek-V2-Lite-Chat, profile_id=metal-default, sample_ids consumed. |
| `training_data/c2_hidden/eagle3_v0/shard_*.parquet` (after running `to-parquet`) | Same data in Parquet for HF datasets / training framework consumption. ~5.7 MB per 1500-record shard with zstd. |
| `models/deepseek-v2-lite-q4.gguf` | Target model (Q4_K_M, ~10 GB). Tokenizer + lm_head weights extracted for shared-decoder use during training. |

Per record, in the parquet:
- `sample_id` (string)
- `pos` (int32) — position within sample (0-indexed)
- `prev_token` (int32) — input token at this position
- `next_token` (int32) — **ground-truth** target (teacher-forced)
- `hidden_f16` (binary, 4096 bytes) — np.frombuffer(..., dtype=np.float16) → shape (2048,)

**Hidden semantics:** the post-final-rmsnorm activation that the target's lm_head consumes to predict the next token. This is the EAGLE-3 distillation signal.

## Target draft-head architecture

EAGLE-3 reference: arXiv 2503.01840, §3.2.

```
Input  (per training step, batch B):
  prev_embed[t]  : Embedding(vocab=102400, dim=2048) lookup of prev_token
  target_hidden[t]: f16[2048]  — captured hidden from data

Concat / project:
  x = Linear(in=2 * 2048, out=2048, bias=False) ([prev_embed; target_hidden])

Transformer block (1 layer, target-aligned shapes):
  x = RMSNorm + MLA-style attention (or Llama-style if simpler) + residual
  x = RMSNorm + SwiGLU MLP (intermediate=5632, matching V2-Lite dense FFN) + residual
  draft_hidden = RMSNorm(x)

Output:
  logits = lm_head(draft_hidden)
    where lm_head is the *frozen target lm_head* (102400 × 2048, shared)
```

**Shapes / counts:**
- Input projection: 2 × 2048 → 2048 = 8.4M params
- Attention block (Q, K, V, O at hidden=2048, 16 heads): ~16M params
- MLP block (gate, up, down at 5632 intermediate): ~35M params
- Output norm: 2048 params
- Total trainable: ~60M params
- File size at fp16: ~120 MB. (EAGLE-3's smaller variants get to ~50 MB by reducing intermediate width — a tunable.)

**Frozen:** target's lm_head + tokenizer embeddings.

## Training hyperparameters (MLX on M3 Pro)

| Param | Value | Justification |
|---|---|---|
| Framework | MLX (Apple Silicon native) | Local-stack decision; see README.md for rationale |
| Optimizer | AdamW | Standard for transformer training |
| LR | 3e-4 | EAGLE-3 paper, Vicuna-class targets |
| LR schedule | cosine, 5% warmup | "" |
| Batch B (sequences) | 16 | Fits 18 GB unified at hidden=2048; halve if OOM during first run |
| Seq length S per batch | 16 (effective 256 positions/step) | EAGLE-3 head sees each (prev, hidden) independently; (B,S) is just a vector-batching convenience. Larger S spends more memory for the same training signal |
| Gradient accumulation | 1 (M3 memory permitting; raise to 2-4 if OOM) | Effective batch 16-64 |
| Epochs | 3 | EAGLE-3 paper. ~600K records / 256 per step = ~2.3K steps/epoch, ~7K total |
| Loss | CE(logits, next_token) | Standard distillation w/ frozen lm_head |
| Auxiliary loss | 0.1 × MSE(draft_hidden, target_hidden) | EAGLE paper §3.3 — drives hidden-geometry alignment for multi-step stability |
| Mixed precision | bf16 trainable / fp16 frozen | M3 Pro supports bf16; matches typical EAGLE-3 setup |
| Wall time @ 5K samples | **~10-15 min on M3 Pro** (1 epoch) | **MEASURED 2026-05-16 morning**: 198 ms/step at B=16 S=16 = ~1294 records/s warm under MLX. 486K records / 1294 ≈ 6 min synthetic; expect 10-15 min with parquet I/O |
| Wall time @ 55K samples (3 epochs) | **~10 hr on M3 Pro** | Linear extrapolation: ~5M records × 3 epochs / 1294 records/s ≈ 3.2 hr compute + ~2-3× overhead for I/O + optimizer = ~10 hr realistic |
| Wall time on H100 (if ever pivoted) | ~1-2 hr at 5K, ~10-20 hr at 50K (paper) | No longer the bottleneck; local M3 Pro fits 55K × 3 epochs within the long weekend |

## Data scale tradeoffs

EAGLE-3 paper Table 6 (acceptance vs training data size on Vicuna-7B):

| Samples | token+1 accept | token+2 accept | Notes |
|---:|---:|---:|---|
| 5K | ~52% | ~28% | What this session captured. Marginal but trains. |
| 50K | ~71% | ~46% | "Useful" regime |
| 500K | ~78% | ~55% | EAGLE-3 published recipe |

**Recommendation:** if H100 time is cheap (e.g. spot $1-2/hr), capture-extend to 50K samples and train at that scale. The 5K shipped this session is the proving-out scale, not the production scale.

## Capture extension protocol (if more data is wanted before training)

```bash
# Extend the same shard with more UltraChat samples (resume-capable).
PY=python3
$PY tools/training/capture_hidden.py prep \
  --out tests/data/ultrachat_50k.jsonl \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --streaming \
  --n 50000 \
  --id-prefix ultrachat \
  --force

./target/release/dismantle capture-hidden \
  --weights models/deepseek-v2-lite-q4.gguf \
  --samples tests/data/ultrachat_50k.jsonl \
  --out training_data/c2_hidden/eagle3_v0/shard_001.bin \
  --max-tokens 128 \
  --no-lm-head \
  --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json
```

Wall-clock @ ~30 records/sec (with `--no-lm-head`):
- 50K × 100 tokens = 5M records → ~46 hr on M3 Pro
- 100K × 100 tokens = 10M records → ~92 hr
- Realistic single-machine ceiling: **~50K samples ≈ 2 days wall**

If capture wall time is the binding constraint, the next-most-impactful engineering investment is `forward_tokens_batched_with_hidden_for_test` (deferred from this session) — extracts per-position hidden from the existing batched-TCB path, ~10-20% throughput improvement over the single-token path.

## Acceptance regression test (post-training)

Once a draft head is trained, sanity-check it before any engine wire-up:

```bash
# Held-out 100-prompt suite — generated, not from training data.
$PY tools/training/eval_acceptance.py \
  --weights models/deepseek-v2-lite-q4.gguf \
  --draft-head models/eagle3-v0.gguf \
  --eval-prompts tests/data/spec_decode_eval_100.jsonl \
  --max-tokens 128 \
  --report reports/path_to_90/stage3_c2/acceptance_eagle3_v0.json
```

(`tools/training/eval_acceptance.py` does NOT exist yet — it's a followup from this session. The script runs target-greedy and then compares to draft-head's top-1 / top-3 / top-5 predictions per position; emits per-position acceptance rates.)

**Pass bar:** ≥ 40% token+1 acceptance on the held-out slice. Below this threshold, retraining with more data is the first move; switching architecture (to MTP-style) is the second.

## Engine wire-up (C3, post-training)

Already sketched in `stage3_c1/architecture.md` §"Engine-integration footprint". The trait seam (`forward_token_with_hidden_for_test`) shipped in this session's commit — C3's only remaining engine work is the `DraftSpecDecoder` module that consumes the trained head + the existing `forward_tokens_batched_for_test` verify path.

A skeleton-stub of `DraftSpecDecoder` is also shipped this session (feature-flagged, no-op draft head, bit-identical to greedy). C3 fills in the trained-head consumer.

## What this brief does NOT cover

- **Path B verify-kernel rewrite** (parallel-K MLA / lm_head / MoE-gate). Per `stage3_spec/audit.md`, even a perfect EAGLE head delivers only ~1.25× e2e at K=4 against today's verify cost. Path B is multi-week kernel work and runs in parallel with the training session.
- **Cost estimation for non-spot H100 rentals.** Spot pricing changes; estimate per current market.
- **Tokenizer/prompt-template alignment for chat-distilled training.** UltraChat is a single-turn user-message dataset; EAGLE-3 paper trains on multi-turn ShareGPT. If acceptance regresses on multi-turn evals, mix in ShareGPT.
