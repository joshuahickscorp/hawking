# Colab notebooks for dismantle calibration

Big-GPU calibration work that doesn't fit on M3 Pro 18 GB.

## Active notebooks

### `qwen3b_mega_calibration.ipynb` ⭐ current focus

**Single Colab run produces calibration data for 4 downstream dismantle projects:**

| Output | Used by |
|---|---|
| Per-prompt parquet shards: tokens + layer-32 residual + intermediate | Eagle5 v2 head training |
| Top-100 logits per token | Quality benchmarks ground truth |
| Per-site activation aggregates (mean/max per channel × 36 layers × 7 sites) | AWQ smoothing, per-channel W4A8 calibration, SmoothQuant |

**Compute:** ~6-8 hr on H100, ~$8-12 in Colab Pro compute units (fits monthly Pro budget).

**Launch:** Open in Colab via `File → Open notebook → GitHub`:
```
https://colab.research.google.com/github/joshuahickscorp/dismantle/blob/main/colab/qwen3b_mega_calibration.ipynb
```

Set GPU: `Runtime → Change runtime type → A100 GPU` (or H100 if you have Pro+).

| GPU | Strategy | Batch | Wall |
|---|---|---|---|
| Blackwell 102 GB | fp16, batch=8 | 8 | ~3 hr |
| A100 80 GB / H100 | fp16, batch=8 | 8 | ~4 hr |
| A100 40 GB | fp16, batch=6 | 6 | ~5 hr |
| L4 24 GB | 4-bit nf4, batch=4 | 4 | ~7 hr |

## After calibration completes (laptop-side work)

Once `qwen3b_corpus/` is on Drive (~3-5 GB), download to laptop and run locally:

```bash
# 1. Train Qwen-3B Eagle5 head (MLX, ~2 hr)
python3 tools/training/eagle5_train.py \
  --corpus-dir artifacts/calibration/qwen3b_corpus \
  --frozen     <qwen3b_frozen_baseline>.npz \
  --ckpt-dir   checkpoints/eagle5_qwen3b \
  --epochs 8 --batch-size 24 --lr 1e-3 \
  --max-rows 4000 --max-row-tokens 128 \
  --sparsity-head proxy --capture-layer 32

# 2. Apply AWQ algorithm to activation aggregates (~30 min, CPU)
python3 tools/training/awq_calibrate.py \
  --stats artifacts/calibration/qwen3b_corpus/per_site_activation_stats.npz \
  --out   profiles/qwen3b_awq_smoothing.json

# 3. Bench stacked configs
DISMANTLE_QWEN_AWQ_SMOOTHING=profiles/qwen3b_awq_smoothing.json \
DISMANTLE_QWEN_W4A8=1 \
EAGLE5_HEAD=checkpoints/eagle5_qwen3b/head_final.safetensors \
TRIALS=10 TOKENS=64 \
  ./tools/bench/eagle5_paired_bench.sh
```

## Expected results stack

| Config | Qwen-3B dec_tps | Comment |
|---|---|---|
| Today (predec default-on) | 26.6 | Current headline |
| + AWQ → W4A8 default-on | ~36 | Quality unblocked |
| + Eagle5 (Qwen-3B head, τ ≈ 3.5) | ~60-80 | Stacked win |

Past llama.cpp's ~50 dec_tps on M3 Pro.

## Historical context

The V2-Lite notebook (`eagle5_v2_corpus.ipynb`) was the original proof-of-concept. It produced 89.20% K=4 acceptance on V2-Lite via `proxy + lr=1e-3` (grid search). That methodology proved the playbook works; this notebook applies it to the actual product target (Qwen-3B) with broader captures (AWQ + quality benchmarks bundled).

The V2-Lite artifacts have been removed since the corpus + trained heads are already on local disk (`artifacts/calibration/v2_lite_corpus/` and `checkpoints/eagle5_v2_*/`).

## Resume behavior

`mega_calibrate.py --skip-existing`-style logic is built in: rerun the same cell after any disconnect and it resumes from the next un-built shard. Drive persistence is the safety net.
