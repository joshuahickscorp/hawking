# tools/training/

Offline scripts that generate training corpora and Eagle5 speculative-decode heads. All outputs land in `../../artifacts/` (gitignored). Nothing here ships in any binary.

## Layout

```
training/
├── build_corpus.py          # corpus of per-layer intermediates from V2-Lite
├── eagle5_train.py          # trains an Eagle5 spec-decode head
├── eagle5_quantize.py       # quantizes a trained head
├── eagle5_tau_eval.py       # evaluates tau / accept rate for a trained head
├── awq_calibrate.py         # AWQ calibration scales
└── requirements.txt
```

## build_corpus.py

Runs sequences of an open chat corpus through DeepSeek-V2-Lite (HuggingFace fp16) and captures per-token, per-layer intermediates:

- `residual_in` — hidden state entering each layer
- `expert_idx` — top-K routed expert IDs per token
- `routing_logits` — pre-softmax router scores
- `intermediate` — gate-up post-SiLU activations (1408-dim per expert)
- `h_high` — residual stream after the MoE block
- `output_logits` — final LM head logits

Output: `artifacts/calibration/v2_lite_corpus/shard_*.parquet` (~27 GB int8-quantized for 10k seqs).

The HuggingFace forward is used because dismantle's hot path deliberately avoids materializing per-layer intermediates through fused kernels. This script runs once, offline.

```sh
# Install deps (once)
python3 -m venv .venv && source .venv/bin/activate
pip install -r tools/training/requirements.txt

# Build corpus (~hours; idempotent per shard, resumes on crash)
python3 tools/training/build_corpus.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --dataset HuggingFaceH4/ultrachat_200k \
    --max-sequences 10000 \
    --out artifacts/calibration/v2_lite_corpus
```

Plan for ~35 GB free under `artifacts/`.

## Notes

- Default backend: HuggingFace transformers + MPS (Apple Silicon GPU via PyTorch).
- If MPS is too slow, fallback: extend `dismantle bench` with `--dump-intermediates PATH` to hook the existing forward path at well-defined points (attn-block-out, moe-block-out, post-RMSNorm, post-LM-head) and write shards from Rust. ~few hundred lines on the model layer.
