# RWKV-7 Micro-Frontier Handoff - 2026-06-20

## Objective

Find the smallest custom RWKV-7 draft that maximizes real speculative decoding throughput. We are no longer treating 100M+ as the center of gravity; those are controls. The real search is the micro-frontier below the old 35M floor.

Spec-decode winner is not "largest" or "lowest loss." It is the smallest/fastest draft that earns enough acceptance:

```text
effective spec throughput = target speed + draft speed + accept rate + verify cost
```

## Current State

- Heavy sweep was cut while running `draft_150m`; completed useful controls already include `35M`, `50M`, `75M`, and `100M`.
- Bucketed fixed-shape batching is implemented and validated; memory no longer climbs from one-off `[B,T]` shapes.
- New micro variants are defined in `tools/training/rwkv7_custom_configs.py`:
  - `draft_17m_probe`: 17.074M, n=128, layers=2, ff=256
  - `draft_18m_probe`: 18.158M, n=128, layers=4, ff=1024
  - `draft_20m_probe`: 19.539M, n=128, layers=8, ff=1024
  - `draft_26m_probe`: 25.919M, n=192, layers=2, ff=512
  - `draft_29m_probe`: 28.964M, n=192, layers=8, ff=768
- Default sweep in `tools/training/launch_draft_sweep.sh` now targets:

```bash
draft_17m_probe draft_18m_probe draft_20m_probe draft_26m_probe draft_29m_probe draft_35m_probe draft_75m_probe
```

## Local Short Tests Passed

```bash
.venv-rwkv/bin/python -m py_compile tools/training/rwkv7_custom_configs.py tools/training/rwkv7_train_draft.py tools/training/rwkv7_spec_hardening.py tools/training/rwkv7_competitive_scorecard.py
.venv-rwkv/bin/python tools/training/test_rwkv7_batch_equiv.py
.venv-rwkv/bin/python tools/training/rwkv7_train_draft.py --variant draft_17m_probe --device mps --dry-run --max-rows 12 --max-length 128 --batch-size 8 --auto-batch 1 --mem-ceiling-gb 17 --grad-accum 1 --grad-checkpoint 1 --empty-cache-every 5 --log-every 1 --save-every 0 --use-chunked --chunk-size 32 --out artifacts/lowbit_rwkv7/runs/custom_draft_17m_probe_dryrun
```

17M dry-run result:

```text
draft_17m_probe: auto-batch=8, opt=2, final_loss=11.1039, mps=1.9G, OK
```

## Main Run For Claude

Run this in the Claude lane, not here:

```bash
cd /Users/scammermike/Downloads/hawking
AUTO_BATCH=1 MEM_CEILING_GB=17 BATCH_SIZE=16 GRAD_ACCUM=1 GRAD_CKPT=1 EMPTY_CACHE_EVERY=5 LOG_EVERY=5 EPOCHS=1 ACCEPT_SEQS=50 DRAFT_VARIANTS="draft_17m_probe draft_18m_probe draft_20m_probe draft_26m_probe draft_29m_probe" bash tools/training/launch_draft_sweep.sh
```

Then compare against existing controls (`35M`, `75M`, `100M`) using accept rate, draft speed, effective spec TPS, and quality. Cut any micro candidate that is dominated by a smaller/faster neighbor.

## What To Watch

- If 17M gets non-trivial acceptance after KD, it may be the winner because it is the true parameter floor with untied 65k embeddings.
- If 17M/18M are too dumb but 26M/29M get close to 35M acceptance, keep the shallow 192-wide line.
- Do not train 100M+ further unless all micro and 35M/75M candidates fail acceptance after KD.
