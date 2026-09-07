# G002 dense anatomy sweep — lane report

Worktree `g002-dense-sweep-20260907-034940`. Science is on disk. This file is the
owed report. Do not push. Metal was not used.

## Acceptance

- `PYTHONPATH=tools/future /usr/local/bin/python3.12 tools/future/dense_sweep.py --selfcheck` exits 0
- `PYTHONPATH=tools/future /usr/local/bin/python3.12 -m pytest hcli/test_dense_anatomy.py -q -m ""` → 14 passed in 93.58s
- Sweep over the real lake wrote `receipts/future/G002_DENSE_ANATOMY_SWEEP.json`
- Every DENSE body has a row that is an anatomy or a NAMED refusal (never empty)
- Spot-check: Qwen3-0.6B `k_proj` 41.27 reproduced; packed bitnet `pre-quantized` reproduced

## Counts (must sum to DENSE=36)

| status | n |
|---|---:|
| ANATOMY | 34 |
| REFUSED | 2 |
| GUARD_STOP | 0 |
| **sum** | **36** |

DENSE count from `G034_LAKE_CENSUS_FULL.json` is 36. Slug sets match. Peak RSS max 9.57 GiB (cap 20). Swapfile delta 0 on every body. Whole sweep ~70 minutes.

## Refusals (mechanism, not empty)

- `microsoft--bitnet-b1.58-2B-4T@04c3b9ad9361` — payload dtype U8, pre-quantized; spectrum would measure the codebook
- `lerobot--pi0_base@25c379b52ba2` — 767 keys contain `expert` as a submodule name; `dense_anatomy` substring-refuses what the census correctly classified DENSE. 36 census targets, 34 anatomised

## Ledger

`odyssey_ledger.progress` on `receipts/future/G034_ODYSSEY_LEDGER.json` after this lane:

- anatomy OWED 38 → 0 for the lake (36 DENSE + later 2 MoE-float remainder)
- DENSE anatomy 34 MEASURED + 2 REFUSED
- Qwen3-0.6B ebpw still 16.0 from the census receipt (other axes not overwritten)

## Table (status, wall, peak RSS, top organ, deficit)

See `receipts/future/G002_DENSE_ANATOMY_SWEEP.json` `rows[]`. Headline:

- Qwen3-0.6B: k_proj 41.27% (published test 41.37; ordering k > q > o > gate > up holds)
- Qwen3-14B: q_proj 39.97%, 238s, 7.07 GiB
- Qwen2.5-72B: q_proj 26.49%, 796s, 6.60 GiB
- Wan2.2: high_noise_model.self_attn.k 42.08%, 653s, 6.53 GiB
- Phi-4-reasoning: qkv_proj 43.98%
- Mistral-Small-24B: self_attn.q_proj 48.10%

Four bodies carry tied ranks under `resolved_ordering` (the Wan2.2 0.04/0.05 pp error class). Ranking is never a raw sort.

## Files this lane wrote

- `tools/future/dense_sweep.py` — driver: headers estimate, guard per body, `resource_cost`, isolate child, incremental receipt, `--resume`, `--apply-ledger`, `--spot-check`
- `receipts/future/G002_DENSE_ANATOMY_SWEEP.json`
- `tools/future/g004_capability_gate.py` — attempted G004 gate; Metal unusable in this sandbox
- `receipts/future/G004_CAPABILITY_GATE.json` — NOT_MEASURED; `mx.default_device()` said gpu, `metal::load_device` failed
- `receipts/future/G009_MOE_ANATOMY_REMAINDER.json` — Inkling-Small + Qwen3.8-Flash-Next, both sharing DEAD

Did not write: `/Volumes`, `tools/future/dense_anatomy.py`. Did stamp `G034_ODYSSEY_LEDGER.json` anatomy (G002 verify required it).

## What in the G002 contract was wrong

1. `dense_anatomy.py` still says “35 of 56 are DENSE”; census is 36. The 36th is pi0_base.
2. Quoted peaks 6.53 / 6.47 vs this run 6.60 / 6.53 (isolated child includes interpreter).
3. `resource_cost` peak RSS is hawkingd RSS; per-body peak is child `ru_maxrss`.
4. `mx.default_device()==gpu` is not a Metal load check (G004 scar).

## What this sandbox cannot close

G004/G007/G008 need `--profile gate`. G005/G006 are HCLI WorkUnits; this lane names no specimen. G019/G016/G017 already have live lanes — do not duplicate.
