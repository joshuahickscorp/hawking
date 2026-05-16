# MLX EAGLE-3 head training stack for path-to-90 C3

Local M3 Pro training of an EAGLE-3 draft head for DeepSeek-V2-Lite-Chat,
using Apple's MLX framework. Owns the full training stack rather than
renting an H100 — slower but free and reusable, and the M3's unified memory
fits the workload cleanly.

## Status

| File | Status | Tested? |
|---|---|---|
| `extract_lm_head.py` | DONE | ✅ produces `v2lite_frozen.npz` (800 MB), logits sanity-checked against a captured hidden vector |
| `model.py` | DONE | ✅ load + forward + loss + grad smoke-tested under MLX runtime; 59.77M trainable params confirmed |
| `data.py` | DONE | ✅ smoke-tested against live partial shard (486K records, 1900 batches/epoch at B=16 S=16); position mask zeros early-pos band correctly |
| `train.py` | DONE — full AdamW loop + cosine LR + position-weighted CE + MSE auxiliary + checkpointing | ⚠️  loop end-to-end untested (smoke runs forward+backward fine; full multi-step + ckpt save not exercised yet) |
| `eval_acceptance.py` | TODO | — |
| `convert_to_dismantle_head.py` | TODO | — |

**Measured perf (2026-05-16 morning, MLX runtime smoke):**

- 198 ms/step warm at B=16 S=16 = ~1,294 records/sec on M3 Pro
- 5K-shard 1 epoch (486K records): ~6 min synthetic, expect ~10-15 min with I/O
- 55K-shard 3 epochs (~15M records est.): **~10 hr total** (vs the brief's
  conservative 30-100 hr estimate — MLX on Apple Silicon is faster than
  projected for this workload)

Implication: once the 55K capture completes Monday morning, training fits
inside Monday and C3 wire-up can start Tuesday with a fresh trained head.

## Morning work — implementation order

### 1. Install MLX + verify (~5 min)

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
$PY -m pip install --user mlx mlx-lm
$PY -c "import mlx.core as mx; print('mlx ok', mx.__version__ if hasattr(mx,'__version__') else 'unknown')"
```

If `pip install --user mlx` complains about a system Python, use the
framework Python directly:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/pip install mlx mlx-lm
```

### 2. Re-verify `extract_lm_head.py` runs cleanly (~5 sec)

`v2lite_frozen.npz` should already exist (800 MB). If not:

```bash
$PY tools/training/mlx_eagle/extract_lm_head.py --verify
```

Expected output ends with:

```
[verify] layout (hidden @ W): vocab=102400 argmax=21854 (corpus next_token was 1217) max=13.378 min=-17.733
```

The argmax/corpus mismatch is correct — the corpus is teacher-forced.

### 3. Smoke-test `model.py` forward pass (~5 min)

Before writing the dataloader, run a one-line script to confirm the
EagleHead constructs and runs forward:

```python
# scratch.py
import mlx.core as mx
from tools.training.mlx_eagle.model import EagleHeadConfig, load_head_from_npz

cfg = EagleHeadConfig()
head = load_head_from_npz("tools/training/mlx_eagle/v2lite_frozen.npz", cfg)

# Fake batch: 4 sequences of 16 tokens each
prev_tokens = mx.zeros((4, 16), dtype=mx.int32)
target_hidden = mx.zeros((4, 16, 2048), dtype=mx.float32)
logits = head(prev_tokens, target_hidden)
print("logits shape", logits.shape)  # expect (4, 16, 102400)
```

**Likely first-run bugs to expect:**
- `trainable_parameters()` API may have moved between MLX versions. If
  `head.trainable_parameters()` errors, fall back to `nn.value_and_grad`
  with manual key filtering.
- `nn.losses.cross_entropy` may want different reduction string ("mean"
  vs "sum_over_batch_size") — check the version's API.
- Concat-then-Linear (`mx.concatenate` + `nn.Linear`) is standard; if
  shape errors, print intermediate shapes.

### 4. ~~Write `data.py`~~ — DONE (see Status table)

Read `training_data/c2_hidden/eagle3_v0/shard_*.parquet` (after running
`tools/training/capture_hidden.py to-parquet`). Each row has:

- `sample_id` (string)
- `pos` (int32)
- `prev_token` (int32)
- `next_token` (int32)
- `hidden_f16` (binary, 4096 bytes → fp16[2048])

Group by `sample_id`, sort by `pos`, emit batches of shape:

- `prev_tokens` : int32[B, S]
- `target_hidden` : float32[B, S, 2048]  (cast from fp16 at batch time)
- `target_next_tokens` : int32[B, S]

Where B is batch size (suggest 16) and S is sequence length (suggest 16
or 32 — *not* the full 128). Reasoning: each (position) in a sample is
trained independently per EAGLE-3 §3.2 (the head sees one (prev, hidden)
pair at a time), so the (B, S) shape is purely a vector-batch convenience
to feed more positions per gradient step. Drop or pad incomplete seqs.

Use `pyarrow.parquet.ParquetFile.iter_batches(batch_size=...)` for memory-
friendly streaming over the full ~2 GB shard.

### 5. Write `train.py` — Adam loop (~2-4 hr)

Skeleton:

```python
import mlx.optimizers as optim

opt = optim.AdamW(learning_rate=3e-4)
loss_and_grad = nn.value_and_grad(head, eagle_loss_closure)

for step, batch in enumerate(data_iter):
    loss, grads = loss_and_grad(head, batch)
    opt.update(head, grads)
    mx.eval(head.parameters(), opt.state)
    if step % 50 == 0:
        print(f"step {step} loss {loss.item():.4f}")
    if step % 1000 == 0:
        save_checkpoint(head, f"ckpt/step_{step}.npz")
```

Hyperparams (from `training_brief.md`):
- AdamW, lr=3e-4, cosine schedule with 5% warmup
- Batch B=16, S=16 effective seq length per batch (~256 positions/step)
- 3 epochs over the ~600K-record 5K-sample dataset = ~110K steps
- Loss = CE + 0.1 * MSE (auxiliary on draft_hidden vs target_hidden)
- Mixed precision: cast trainable params to bf16, frozen weights stay fp16

Expected wall on M3 Pro with capture NOT running: ~5-10 hr per epoch.
First-run advice: start with 1 epoch (~3 hr) to validate loss decreases,
then commit to 3 epochs.

### 6. Write `eval_acceptance.py` — held-out test (~2 hr)

Take ~100 held-out prompts (NOT in `tests/data/ultrachat_5k.jsonl`), run
target greedy to get reference next-token sequence, then run the EAGLE
head on each position and check top-1 acceptance. Report per-position
acceptance rate. Pass bar (per `architecture.md`): ≥40% token+1.

### 7. Write `convert_to_dismantle_head.py` — MLX → dismantle .gguf (~3-4 hr)

This is the most engine-coupled piece — defer until training works.
Write `models/eagle3-v0.gguf` with tensor layout that the future C3
`EagleDraftHead` impl in `crates/dismantle-core/src/speculate/draft_head.rs`
will consume.

## Memory budget (M3 Pro 18 GB unified)

- Frozen weights : ~840 MB (token_embd + lm_head)
- Trainable     : ~240 MB (fp32 master) + ~480 MB (Adam m + v) = ~720 MB
- Activations   : ~2-3 GB at B=16, S=16
- MLX runtime + Python : ~1-2 GB
- **Total      : ~5-6 GB** — leaves ~12 GB free for OS + browser + IDE

The dismantle inference engine takes ~10 GB when loaded (model + KV).
You CAN'T run capture and training simultaneously — both want the M3's
GPU + ~10 GB unified memory. Run them sequentially.

## Cancel-and-resume

If training crashes or you stop mid-run, restart from the most recent
checkpoint. `convert_to_dismantle_head.py` reads the last `ckpt/step_*.npz`.

If you want to abandon training and re-train from scratch (e.g. after
extending dataset to 50K), just `rm -rf ckpt/` and re-run `train.py`.

## What this does NOT do

- Does NOT train multi-step draft prediction (proposing K>1 tokens via
  recurrent application of the head). EAGLE-3 §3.2 keeps the head
  single-step; multi-step comes from running the head K times serially
  at inference. Defer until C3 wire-up.
- Does NOT use Apple Neural Engine or AMX directly — MLX abstracts the
  hardware choice. If the training is too slow, profiling MLX's choice
  of backend is the first optimization.
- Does NOT validate accept-rate during training — that's `eval_acceptance.py`
  after training completes.
