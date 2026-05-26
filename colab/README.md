# Colab notebooks for dismantle calibration

Big-GPU calibration work that doesn't fit on M3 Pro 18 GB.

## Active notebooks

### `qwen_past_200_h100.ipynb` ⭐ past-200 push

H100-first run-all for the next throughput ceiling break. It trains stacked
Eagle5 candidates for the Qwen2.5-1.5B student path, emits AWQ and Q2/IQ2
calibration artifacts, runs tau/frontier ranking, and writes a
`past200_summary.md` handoff with exact local runtime hints.

Launch:
```
https://colab.research.google.com/github/joshuahickscorp/dismantle/blob/main/colab/qwen_past_200_h100.ipynb
```

### `qwen3b_mega_calibration.ipynb` ⭐ current focus

**Single Colab run produces calibration data for 4 downstream dismantle projects:**

| Output | Used by |
|---|---|
| Per-prompt parquet shards: tokens + layer-32 residual + intermediate | Eagle5 v2 head training |
| Top-100 logits per token | Quality benchmarks ground truth |
| Per-site activation aggregates (mean/max per channel × 36 layers × 7 sites) | AWQ smoothing, per-channel W4A8 calibration, SmoothQuant |

**Compute:** ~4-8 hr depending on GPU, with Drive-backed resume during the run.

**Launch:** Open in Colab via `File → Open notebook → GitHub`:
```
https://colab.research.google.com/github/joshuahickscorp/dismantle/blob/main/colab/qwen3b_mega_calibration.ipynb
```

Set GPU: `Runtime → Change runtime type → A100 GPU` (or H100 if you have Pro+).

| GPU | Strategy | Batch | Wall |
|---|---|---|---|
| G4 / Blackwell / H100 70GB+ | fp16, chunked LM head | 8 | ~3-4 hr |
| A100 40 GB | fp16, chunked LM head | 6 | ~5 hr |
| L4 24 GB | 4-bit nf4, chunked LM head | 4 | ~7 hr |
| T4/V100 16 GB | 4-bit nf4, chunked LM head | 2 | slow but safer |

## After calibration completes (laptop-side work)

Once `qwen3b_corpus/` is on Drive (size depends on actual token lengths;
expect several GB+), download to laptop and run locally:

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

`mega_calibrate.py` resumes from the next contiguous shard found either on
local SSD or Drive. It also saves `per_site_activation_stats.npz` as it goes;
if shards exist but matching stats are missing/stale, the script stops instead
of silently producing bad AWQ/W4A8 calibration data.
