# EAGLE-4

Routing-aware speculative-decoding head for Mixture-of-Experts targets. Trained
and evaluated on **DeepSeek-V2-Lite-Chat Q4_K_M** on Apple Silicon via MLX.

```
eagle4.py     head + train + eval + quantize  (~500 lines)
capture.py    V2-Lite per-layer state → parquet
bench.py      EAGLE-3 baseline trainer + compare
tau_eval.py   τ-at-depth-K eval (the spec-decode-honest metric)
q4_parity.py  Q4 head ↔ bf16 head parity check
```

The runtime that turns offline acceptance into wall-clock tps lives separately
in a Rust+Metal spec-decode runtime — this repo is the head architecture and
training pipeline.

## Results — head-to-head vs EAGLE-3

Both heads trained from the same V2-Lite-Chat Q4_K_M per-token captures
(1M records, one epoch). Same trainer, same loss balance, same held-out
shard (999 windows × 5 contiguous positions = 5,000 tokens, disjoint
conversations). Only difference: architecture.

### τ-at-depth-4 — the metric that translates to wall-clock

| | EAGLE-3 | **EAGLE-4** | Δ |
|---|---|---|---|
| **mean accepted prefix length** ↑ | **2.15** | **3.57** | **+66% relative** |
| full-depth-4 acceptance ↑ | 37.4% | **83.6%** | +46.2 pp |
| depth-1 accept | 73.8% | 95.3% | +21.5 pp |
| depth-2 accept | 57.0% | 90.6% | +33.6 pp |
| depth-3 accept | 46.8% | 87.1% | +40.3 pp |
| depth-4 accept | 37.4% | 83.6% | +46.2 pp |

τ-at-depth-K is the spec-decode-honest number: roll the head autoregressively,
feeding its own argmax as the next `prev_token`, and count how many tokens
get accepted before the head disagrees with V2-Lite. EAGLE-4's τ=3.57 out of
a max-4 means 3.57 tokens accepted per verify step on average — vs EAGLE-3's
2.15. Under greedy spec-decode that's a 1.66× improvement in the dominant
cost term. EAGLE-4's depth-4 acceptance (83.6%) is higher than EAGLE-3's
depth-1 (73.8%): EAGLE-4 still agrees with V2-Lite four tokens deep into
autoregressive rollout more often than EAGLE-3 agrees on the first token.

### Single-step + mask recall

| | EAGLE-3 | EAGLE-4 (best) | EAGLE-4 (routing) |
|---|---|---|---|
| target-argmax acceptance ↑ | 75.84% | **95.32%** | 95.20% |
| corpus top-1 ↑ | 57.12% | 57.04% | 57.44% |
| per-layer mask top-8 recall ↑ | n/a | 17.01% | **21.13%** |
| calibration scalar | no | yes | yes |
| training wall-clock | ~10 min | ~21 min | ~21 min |

`best.npz` is the τ-optimal checkpoint (step 1000). `best_routing.npz` is
the same run stopped at step 1800 — trades 0.01 τ for +4 pp mask recall.
Pick by runtime bottleneck: target-token-throughput → `best.npz`;
expert-prefetch-bandwidth → `best_routing.npz`. EAGLE-3 emits no mask and
no calibration signal — it can't, by design.

### Q4 head

bf16 head is 297 MB. Q4 quantization (group_size 64, matching V2-Lite's own
Q4_K_M) shrinks it to **46 MB (6.40× smaller)** at **99.90% argmax-parity**
with the bf16 source.

Full ablation curve + the negative result on multi-step training in
[bench_results.md](bench_results.md). Design notes in
[ARCHITECTURE.md](ARCHITECTURE.md).

**Not yet measured:** wall-clock tps. Requires the Rust+Metal spec-decode
runtime.

## Reproduce

```bash
pip install -e .

# 1. Pull V2-Lite frozen weights
python eagle4.py frozen --out v2lite_frozen.npz

# 2. Capture training + held-out shards
python capture.py --out-dir data/train   --n-records 1000000 --skip-n 5000
python capture.py --out-dir data/heldout --n-records    5000 --skip-n 0

# 3. Train EAGLE-4 (~21 min on M3 Pro)
python eagle4.py train --parquet data/train/*.parquet \
    --frozen v2lite_frozen.npz --ckpt-dir ckpt/eagle4

# 4. Train EAGLE-3 baseline (~10 min)
python bench.py train --parquet data/train/*.parquet \
    --frozen v2lite_frozen.npz --ckpt-dir ckpt/eagle3

# 5. τ-at-depth-4 head-to-head
python tau_eval.py compare \
    --eagle3 ckpt/eagle3/latest.npz --eagle4 ckpt/eagle4/latest.npz \
    --frozen v2lite_frozen.npz --parquet data/heldout/*.parquet --depth 4

# 6. Q4 quantize + parity
python eagle4.py quantize --in ckpt/eagle4/latest.npz --out ckpt/eagle4/q4.npz
python q4_parity.py ckpt/eagle4/latest.npz ckpt/eagle4/q4.npz \
    v2lite_frozen.npz data/heldout/shard_00000.parquet 1000
```

Total ≈ 60 min wall on M3 Pro 18 GB (capture is the slow part).

## The idea, in one paragraph

EAGLE-1/2/3 are speculative-decoding heads for dense LLMs: they predict the
next token. For Mixture-of-Experts targets, predicting *which experts will
fire* is at least as valuable — a runtime that knows the future routing can
prefetch only those experts and skip the verify-side dequant for the rest.
EAGLE-4 emits both: a token distribution **and** a `[26 layers × 64 experts]`
mask per token, plus a per-token calibration scalar. The architectural trick
is a residual gate around the transformer block, initialized small, so the
head's first prediction is near-identical to V2-Lite's own — training learns
only the small refinement. Full design in [ARCHITECTURE.md](ARCHITECTURE.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
