# Colab Eagle5 v2 corpus build

The DeepSeek-V2-Lite calibration corpus is too big to generate efficiently
on the M3 Pro 18 GB laptop (the 16B model has to be offloaded to CPU/disk,
making forwards 20× slower → ~5-day ETA for 3000 sequences). On Colab GPUs
the same job finishes in 1-3 hours depending on tier:

**Current notebook config:** MAXED — 20,000 sequences × 4,096 max-tokens
(2× the "safe" config). Yields ~+8-12 percentage points on acceptance
rate vs the 10k/2k config; cost is wall time.

| Tier | GPU | VRAM | Corpus 20k seqs × 4k tokens | Notes |
|---|---|---|---|---|
| Free | T4 | 16 GB | **~5-6 hr** | 4-bit load via bitsandbytes; needs keepalive |
| Pro | V100 | 16 GB | **~2 hr** | 4-bit load; 24h sessions, no idle disconnect |
| Pro/Pro+ | A100 | 40 GB | **~1 hr** | Native fp16; batch=16 |

If 5-6 hr on T4 is too long for your window, edit Cell 4 in the notebook
to bring `--max-sequences` back to `10000` and `--max-tokens-per-seq` to
`2048` (the "safe" 2.5-hr config).

## Quick start

1. **Open the notebook in Colab:**
   - Go to https://colab.research.google.com
   - `File → Upload notebook` → pick `colab/eagle5_v2_corpus.ipynb` from this repo
   - OR drop the URL: `File → Open notebook → GitHub` and paste
     `https://github.com/joshuahickscorp/dismantle/blob/main/colab/eagle5_v2_corpus.ipynb`

2. **Enable GPU:** `Runtime → Change runtime type → T4 GPU` (or V100/A100 on Pro).

3. **Run cells 1-4 in order.** Cell 4 is the corpus build (the long step).

4. **Run Cell 5 (keepalive) in a SECOND browser tab.** Free Colab disconnects
   idle tabs after ~90 min. The keepalive prints a heartbeat every 60 sec to
   keep the tab "active." Stop it once Cell 4 finishes.

5. **When Cell 4 completes**, Cell 6 verifies and prints download instructions.

## Resume after disconnect

If Colab kicks you mid-build, just re-run Cell 4. `build_corpus.py
--skip-existing` is on by default — it detects shards already on Drive and
resumes from the next one. **Nothing is lost.**

The corpus output dir is on Google Drive (`MyDrive/dismantle/v2_lite_corpus/`)
not on Colab's ephemeral disk, so shards survive any session disconnect.

## After corpus is built

Download from Drive to laptop, then resume the chain locally:

```bash
# 1. On Drive web UI, right-click MyDrive/dismantle/v2_lite_corpus
#    → Download (Drive zips it, ~1.5 GB compressed)
# 2. Extract on laptop:
cd ~/Downloads/dismantle/artifacts/calibration/
unzip ~/Downloads/v2_lite_corpus-*.zip
# Move shards into the expected dir
mv v2_lite_corpus/*.parquet artifacts/calibration/v2_lite_corpus/

# 3. Resume the overnight chain — corpus step skips (shards exist) so it
#    goes straight to train (~1.5 hr on M3 Pro with MLX) + eval + bench.
tools/bench/overnight_eagle5_2026_05_26.sh
```

## Why not train on Colab too?

Eagle5 training uses MLX (Apple-only), which doesn't run on CUDA. Porting
to PyTorch is possible but the head is tiny (~10 MB params), so MLX on M3
Pro finishes in ~1.5 hr — faster than PyTorch on T4 would be once you factor
in the corpus download wait. Train locally; only offload the GPU-heavy
corpus generation to Colab.

## Troubleshooting

**"No CUDA GPU"** in Cell 1: You didn't enable GPU. `Runtime → Change runtime
type → T4 GPU`.

**`bitsandbytes` import error:** Restart runtime after Cell 2: `Runtime →
Restart runtime`. The bnb install needs a fresh Python.

**Cell 4 OOMs:** You got a smaller-VRAM GPU than expected. Drop `--batch-size`
to 2 in the notebook (edit the `batch` variable). Or restart and hope for
a better assignment.

**Idle disconnect during corpus:** Re-run Cell 4. Skip-existing picks up
where it left off. To prevent in future: run Cell 5 in a second tab.

**Drive auth expired:** Re-run Cell 1.
