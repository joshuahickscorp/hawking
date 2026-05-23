# tools/training/

Offline scripts that generate artifacts consumed by Phase 1 levers in
`dismantle-execution-plan-enchanted-salamander.md`.

```
training/
├── build_corpus.py    # one-shot V2-Lite forward → per-layer intermediates
├── eagle5_train.py    # (queued, task #13) — trains the activation predictor
└── README.md          # this file
```

All outputs land in `../../artifacts/` (gitignored). Nothing in this
directory ships in any binary — these are bench/calibration tools.

## build_corpus.py

Runs ~10k sequences of an open chat corpus through DeepSeek-V2-Lite (the
HuggingFace fp16 release, not the GGUF) and captures per-token, per-layer:

- `residual_in`        — hidden state entering each layer
- `expert_idx`         — top-K routed expert IDs per token
- `routing_logits`     — pre-softmax router scores
- `intermediate`       — gate-up post-SiLU activations (1408-dim per expert)
- `h_high`             — residual stream after the MoE block
- `output_logits`      — final LM head logits

Output: `artifacts/calibration/v2_lite_corpus/shard_*.parquet` (~27 GB
int8-quantized total).

Why a separate forward, not dismantle's own engine: dismantle's hot path
deliberately doesn't materialize per-layer intermediates (they go through
fused kernels). Adding intermediate capture would either pull data off the
GPU on every layer (bandwidth-hostile, slow) or require a parallel
debug-mode forward. The HuggingFace fp16 forward is slow but trivial to
hook — and this script runs once, offline, async to dev work.

**If transformers/MPS turns out too slow on the target M3 Pro:** the fallback
is to extend `dismantle bench` with `--dump-intermediates PATH`, hook the
existing forward path at well-defined points (attn-block-out, moe-block-out,
post-RMSNorm, post-LM-head), and write shards from there. ~few hundred lines
of Rust on the model layer. Queue as a new task if needed.

## Backend choice

Default: **HuggingFace transformers + MPS** (Apple Silicon GPU via PyTorch).

| Backend | Pros | Cons |
|---|---|---|
| HF transformers + MPS | hooks are first-class; well-documented; output_hidden_states + output_router_logits supported | slowest (~hours per 10k seqs); needs HF auth for V2-Lite weights |
| MLX-LM | Apple-native, faster than MPS | intermediate-capture hooks are harder to wire (manual model edit) |
| llama-cpp-python | matches the GGUF in use | no clean MoE intermediate capture in upstream |
| dismantle `--dump-intermediates` | reuses the optimized engine | requires Rust-side instrumentation (multi-session) |

## Running

```bash
# 1. Install deps (once)
python3 -m venv .venv && source .venv/bin/activate
pip install -r tools/training/requirements.txt

# 2. Build the corpus (long-running; runs to background OK)
python3 tools/training/build_corpus.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --dataset HuggingFaceH4/ultrachat_200k \
    --max-sequences 10000 \
    --out artifacts/calibration/v2_lite_corpus

# 3. Verify shape (quick sanity check)
python3 -c "import pyarrow.parquet as pq; \
    t = pq.read_table('artifacts/calibration/v2_lite_corpus/shard_0000.parquet'); \
    print(t.schema); print(t.num_rows)"
```

Re-runs are idempotent per-shard: existing shards are skipped, so a
crashed run resumes from where it left off.

## Disk budget

~27 GB int8-quantized for 10k seqs × avg 1k tokens × 27 layers ×
(residual_in 2048 + routing_logits 64 + intermediate 1408 × top-K + ...).
Plan for 35 GB free under `artifacts/` to be safe.
