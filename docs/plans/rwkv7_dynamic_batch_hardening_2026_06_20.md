# RWKV-7 Dynamic Batch Hardening - 2026-06-20

## Findings

- The fixed fast preset (`BATCH_SIZE=16`, `GRAD_CKPT=0`, old 0.9 MPS cap) was unsafe at real `max_length=1024`: deep variants OOMed immediately.
- The auto-batch path fixed the first class of OOM by probing under `MEM_CEILING_GB`, but the probe only halved after OOM. That under-sized some models and did not actually find the largest fitting batch.
- The later 35M failure was different: it reached opt step 150, then OOMed in backward with `other allocations: 10.02 GiB`. That points to retained MPS cache growth from `EMPTY_CACHE_EVERY=0`, not bad dynamic batch sizing.
- The runtime OOM guard only wrapped forward/loss. The observed failure happened in backward, so it escaped the guard and killed the sweep.
- MPS also retains resources per distinct `[B,T]` shape. Padding batched sequence lengths to a 256-token grid bounds shape churn, but the dynamic packer must budget by that rounded padded length or it can over-pack short examples.
- Follow-up diagnostic showed flushing every optimizer step still climbed monotonically (`6.3G -> 7.3G -> 9.3G -> 10.9G -> 12.4G -> 13.9G -> OOM`). That made the final issue structural: variable token-budget batches produced a near-new `[B,T]` shape every step and Metal retained per-shape resources in `other allocations`.
- The launcher help still advertised the old unsafe preset, which made it easy to relaunch the failing mode.

## Implemented

- `probe_max_batch` now binary-searches the true largest full-length batch that fits, instead of halving down to a conservative value.
- MPS cache flushing is centralized in `flush_mps_cache`, with guarded synchronize/empty-cache calls for post-OOM recovery.
- Direct trainer default for `--empty-cache-every` is now 5, matching the safe sweep default.
- The SFT loop now catches OOM from forward or backward, clears partial gradients, flushes MPS, splits the current batch group into smaller groups, and retries instead of crashing or dropping data.
- Batched SFT now pads to a 256-token length grid and `AUTO_BATCH=1` uses fixed-B length buckets (`bucketed_fixed_groups`) so the distinct MPS shapes are bounded.
- Progress logging uses a shared MPS driver-memory helper and includes `mps=X.YG` when available.
- Launcher help now recommends the safe fast preset:

```bash
AUTO_BATCH=1 MEM_CEILING_GB=17 BATCH_SIZE=16 GRAD_ACCUM=1 GRAD_CKPT=1 EMPTY_CACHE_EVERY=5 LOG_EVERY=5 bash tools/training/launch_draft_sweep.sh
```

## Claude Prompt

Please validate the RWKV-7 draft sweep hardening. Focus on `tools/training/rwkv7_train_draft.py` and `tools/training/launch_draft_sweep.sh`.

Context: the old speed preset OOMed immediately on deep variants. Auto-batch then worked but only used halve-on-OOM probing. A production 35M run reached opt step 150 and failed in backward due retained MPS cache growth when `EMPTY_CACHE_EVERY=0`.

What changed: true binary-search max-fit probe, default periodic MPS cache flush every 5 optimizer steps, guarded MPS flush helper, runtime OOM catch around SFT forward/backward that splits the current group and retries, and safer launcher help.

Smoke checks already passed:

```bash
python3 -m py_compile tools/training/rwkv7_train_draft.py tools/training/test_rwkv7_batch_equiv.py
.venv-rwkv/bin/python tools/training/test_rwkv7_batch_equiv.py
.venv-rwkv/bin/python tools/training/rwkv7_train_draft.py --variant draft_35m_probe --device mps --dry-run --max-rows 12 --max-length 128 --batch-size 4 --auto-batch 1 --mem-ceiling-gb 17 --grad-accum 1 --grad-checkpoint 1 --empty-cache-every 5 --log-every 1 --use-chunked --chunk-size 32 --out artifacts/lowbit_rwkv7/runs/custom_draft_35m_probe_dryrun
```

Variable-shape validation passed the first old failure point but was not enough by itself:

```text
draft_35m_probe: auto-batch=8, opt=175, final_loss=7.1526, tok/s=3198, mps=8.8G, hours=0.05
Previous failure was opt=150 in backward with max allowed 15.83 GiB.
```

After switching `AUTO_BATCH=1` to fixed-B length buckets, the decisive memory diagnostic passed beyond the later step-180 failure:

```text
draft_35m_probe: auto-batch=8, opt=220, final_loss=6.5903, tok/s≈3510, mps=8.4G at opt=200, hours=0.05
Memory curve: opt25 9.0G, opt50 10.1G, opt75 9.5G, opt100 8.2G, opt125 8.3G, opt150 8.3G, opt175 8.3G, opt200 8.4G.
This replaces the bad curve that climbed monotonically to OOM.
```

Command used:

```bash
.venv-rwkv/bin/python tools/training/rwkv7_train_draft.py --variant draft_35m_probe --device mps --epochs 1 --max-steps 220 --batch-size 16 --auto-batch 1 --mem-ceiling-gb 17 --grad-accum 1 --grad-checkpoint 1 --empty-cache-every 5 --log-every 25 --save-every 0 --use-chunked --chunk-size 32 --out artifacts/lowbit_rwkv7/runs/custom_draft_35m_probe_bucketed_validate
```

Please still check for logic bugs in gradient accumulation after an OOM split before launching the full 7-variant unattended sweep.
