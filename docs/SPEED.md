# `speed` branch — RWKV-7 training throughput + max-RAM

Branched from `main` @ the post-rename clean slate. Two things live here:
1. A **~16× faster** RWKV-7 draft trainer that uses RAM for speed (opt-in, bit-exact).
2. A **manifest** of the stale unmerged feature branches (consolidation status + hazard).

## 1. Training speed-up (measured)

A/B on the 50M draft config, same 16,384-token budget (measured *under live-sweep GPU
contention*, so absolute tok/s is depressed but the ratio holds):

| setting | tok/s | peak RAM |
|---|---|---|
| **current** — bs=1, `empty_cache()` every example, grad-checkpoint on | 64 | ~1.2 GB |
| **optimized** — bs=16, no `empty_cache()`, grad-checkpoint off | 1017 | ~7.8 GB |
| | **~16×** | **6.4× RAM** |

Three independent levers, all in `tools/training/rwkv7_train_draft.py` (all opt-in;
defaults reproduce the old behaviour exactly):

1. **Batching (`--batch-size N`)** — the old loop ran one sequence per forward/backward
   (`batch=1` + grad-accum). Now `N` sequences are right-padded into one `[N, T]` forward.
   RWKV-7's recurrence is strictly left-to-right, so right-padding cannot perturb real
   positions — **verified bit-exact** (`tools/training/test_rwkv7_batch_equiv.py`, worst
   |Δlogit| = 0.0 across chunked + non-chunked paths). This is the bulk of the win and the
   main RAM consumer.
2. **No per-example `empty_cache()`** — the old loop called `torch.mps.empty_cache()` after
   **every example**, which returns memory to the driver and serialises MPS each step (and
   pins RAM low). Now it is off by default (`--empty-cache-every 0`); set N>0 only as a
   safety valve.
3. **Grad-checkpoint off (`--grad-checkpoint 0`)** + **`--mps-mem-fraction 0.9`** — skip
   activation recompute (faster, more RAM) and let MPS use up to 90% of unified RAM.

### Run a fast sweep

Same *effective* batch as the current default (so training dynamics are preserved —
`BATCH_SIZE*GRAD_ACCUM` held at 16), just far faster:

```bash
BATCH_SIZE=16 GRAD_ACCUM=1 GRAD_CKPT=0 MPS_MEM_FRACTION=0.9 \
DRAFT_VARIANTS="draft_35m_probe draft_50m_probe draft_75m_probe draft_100m draft_150m draft_200m draft_300m" \
DRAFT_EPOCHS=1 USE_CHUNKED=1 SEED=1337 \
  nohup caffeinate -dimsu bash tools/training/g1a_v2_expansion_chain.sh 3.4489 pass \
  > artifacts/lowbit_rwkv7/master_chain.log 2>&1 &
```

To push RAM toward 90% (≈16 GB on the 18 GB box) for the larger variants, raise
`BATCH_SIZE` (e.g. 32) and lower `GRAD_ACCUM` to match; `MPS_MEM_FRACTION=0.9` caps it
safely. Bigger `BATCH_SIZE` changes the effective batch (a mild dynamics change), unlike the
16×1 preset which is dynamics-neutral.

> NOTE: batching changes training results vs `batch=1` (token-level vs example-level loss
> mean), so a **fast sweep should start fresh** for consistent cross-variant accept-rate
> comparison — don't splice it into a `batch=1` run mid-flight.

## 2. Unmerged-branch consolidation — manifest + hazard

⚠️ The 8 unmerged feature branches **cannot be safely merged here**: they branched ~4 days
ago and are **37-40 commits behind main, predating the dismantle→hawking rename**. They
still contain the old `crates/dismantle-*` tree, so merging them would **resurrect every
`dismantle` name** we just removed and conflict massively. They are intentionally **NOT
merged** into `speed`. Bringing a branch forward means rebasing/cherry-picking its unique
commits onto the renamed tree (per-branch conflict resolution) — decide per branch:

| branch | +commits | what it adds | suggested disposition |
|---|---|---|---|
| `rwkv7/chunked-scan` | 14 | DPO pair-builder + chunked-scan (core already in main) | cherry-pick DPO bits if wanted; else delete (superseded) |
| `rwkv7/multiseq-inspect` | 8 | multiseq continuous-batch decode + Gate-2 fix + fmt | rebase-forward if pursuing batched decode |
| `rwkv7/torch-trainer` | 7 | SimPO tuning, supervised-only loss, 19 GB-fit | mine for ideas; loss-opt already in main trainer |
| `rwkv7/multiseq` | 6 | slot-major continuous-batch decode dispatcher | superseded by `multiseq-inspect` — delete |
| `posttrain-pipeline` | 3 | batched teacher capture + streaming trainer (~3× KD) | rebase-forward — genuinely useful for KD speed |
| `rwkv7/lora-fuse-recovered` | 2 | LoRA-GEMV fusion in multiseq path | rebase-forward only with multiseq |
| `ci/fmt-rwkv-tokenizer` | 1 | rustfmt of the World tokenizer | likely already in main — delete |
| `rwkv7/world-tokenizer` | 1 | World tokenizer (greedy-trie) | already in main — delete |

Recommendation: delete the 4 superseded/already-in-main branches, and (on request)
rebase-forward `posttrain-pipeline` + the multiseq trio onto current main one at a time.
