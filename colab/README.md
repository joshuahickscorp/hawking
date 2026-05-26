# Colab notebooks for dismantle calibration

Big-GPU calibration work that doesn't fit on M3 Pro 18 GB.

## Active notebook

### `qwen3b_reconciliation.ipynb` ⭐ current

Unified successor to `qwen3b_full_stack.ipynb` and the now-retired
`qwen_past_200_h100.ipynb` (kept under `legacy/`). One run-all that:

1. Builds/resumes 10k-seq corpora for Qwen2.5-3B-Instruct and Qwen2.5-1.5B-Instruct.
2. Computes both **per-tensor** AWQ smoothing (heuristic alpha=0.5) and
   **per-channel adaptive** AWQ smoothing (`colab/awq_per_channel_calibrate.py`).
3. Emits Q2/IQ2 importance artifacts for the future sub-2-bit ship.
4. Runs an **Eagle6 grid sweep** (5 specs × 1-block and 2-block × LR/residual-delta
   variants) per target, then **one extra-long training session** on the per-target
   grid winner (20 epochs, full corpus, 192-token windows).
5. Runs tau ranking + frontier policy search on the long-trained head.
6. **Simulates spec-decode acceptance in-notebook** against held-out corpus
   windows — gives an honest upper bound BEFORE the Mac runtime port lands.

Launch:
```
https://colab.research.google.com/github/joshuahickscorp/dismantle/blob/main/colab/qwen3b_reconciliation.ipynb
```

**Compute:** A100-40GB at MAX_QUALITY_MODE: ~4-6 hr (corpus build + AWQ + grid +
long-train + tau + frontier + simulation × 2 targets). Drive-backed resume; safe
to interrupt and re-run.

### Why "reconciliation"?

The May 2026 end-to-end paired bench discovered that `--speculate eagle5` on
Qwen-3B/1.5B is a no-op: spec-decode is wired into `deepseek_v2.rs` only, not
`qwen_dense.rs`. The trained heads from the prior notebooks were sitting unused
in RAM. This notebook produces the **best possible heads** (Eagle6 with all the
levers we've accumulated) so they're ready inventory once the Rust port lands.

See `docs/eagle5_qwen_port_plan.md` for the local Rust port spec.

## Levers preserved from the retired `qwen_past_200_h100.ipynb`

- Variable hidden size (Qwen2.5-1.5B vs 3B)
- `--num-blocks` (1-block and 2-block Eagle heads)
- `--head-heads` and `--head-ff-mult`
- Q2/IQ2 importance calibration
- 1.5B student path

## Levers added in this notebook

- **Per-channel adaptive AWQ** — channels with higher activation magnitude get
  higher alpha, smoothing the outliers more aggressively without over-smoothing
  the quiet channels.
- **Extra-long training session** — winner spec retrained for 20 epochs on the
  full 10k corpus with 192-token windows and tuned residual-delta + calib losses.
- **In-notebook spec-decode simulation** — Python-side replay of the trained
  head's drafts against held-out corpus, returns per-step accept rates so we
  know if the head is good *before* a 2-4 day Mac port.

## After Colab completes

Download the long-trained head from Drive to the Mac project tree:

```bash
# Example for 3B path:
gdown <drive-url> -O checkpoints/eagle6_q3b_long/head_final.safetensors

# Same for 1.5B if you want both:
gdown <drive-url> -O checkpoints/eagle6_q1p5_long/head_final.safetensors
```

The reconciliation summary on Drive (`reconciliation_summary.md`) contains the
exact head paths and the runtime hints (lattice K/width, entropy threshold,
variable-K conf thresh) the future Mac runtime port should consume.

## Sibling notebooks

### `qwen3b_full_stack.ipynb` (predecessor)

Still works; produces Eagle5 (not Eagle6) heads on a single Qwen-3B target with
a fixed grid. Use this if you want a faster (~90 min) single-target run without
the per-channel AWQ + long-train + simulation passes.

### `qwen3b_mega_calibration.ipynb` (calibration-only)

Drops out the training entirely — just builds the corpus + AWQ + frozen
baseline. Useful when you want the calibration artifacts but plan to train
locally.

### `legacy/qwen_past_200_h100.ipynb` (retired)

Kept for reference. The 14-spec grid (1.5B + 3B × 7 specs each) it ran was the
right direction but didn't ship the long-train pass or in-notebook simulation.
The reconciliation notebook supersedes it.

## Resume behavior

Every long-running step in the reconciliation notebook checks for its output
artifact before launching the subprocess. Set `FORCE_*=True` in Cell 1 to bust
a specific cache. Corpus shards are Drive-backed at `--sync-every 4` so a
Colab disconnect mid-build only loses a few shards.
