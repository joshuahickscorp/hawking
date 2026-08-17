# Genesis children: measured capacity and the pool

Qwen3.8 (uniform-q4-v1) now has a spawn/poll/kill pool. The first question
was empirical: how many children actually fit?

## What is shared, and what is not

The 8.5 GB artifact is mmap'd. That does **not** help. Each child copies the
catalog into a **private** Metal `IOAccelerator (graphics)` buffer:

| n | seq-len | phys_footprint each | IOAccelerator dirty | mapped-file clean |
|---|--------:|--------------------:|--------------------:|------------------:|
| 1 | 128     | 15.659 GB           | 14.476 GB           | 4.4 MB            |
| 1 | 8192    | 16.721 GB           | 15.533 GB           | 4.4 MB            |
| 2 | 128     | 15.663 GB × 2       | 14.476 GB × 2       | 4.5 MB × 2        |
| 4 | 2048    | 15.96 GB × 4        | 14.728 GB × 4       | 4.5 MB × 4        |

IOAccelerator dirty is bit-identical across children at the same seq-len.
mapped-file is ~4.5 MB, not 8.5 GB. The naive `96 / 15.5 = 6` was wrong in
the "fewer fit" direction.

**Trust `footprint` `phys_footprint` and machine `wired + anonymous +
compressor`.** `ps` RSS both undercounts Metal (5.9–12.9 GB vs 15.96 GB phys
at N=4) and would overcount shared file pages if the artifact were mapped.

## Seq-len does not double N

KV is `16 layers × 2 (K,V) × 4 heads × 256 × 4 B = 131,072 B` per position
(source constants). 8192 − 128 = 1,056,964,608 B.

Measured IOAccelerator delta, N=1: `15,533,162,496 − 14,476,197,888 =
1,056,964,608 B`. Exact match. Complete-token wall went 37.855 ms → 40.996
ms. Search tasks stay at `--max-seq-len 128`.

## Throughput knee is N=2

Complete-token wall (DIRTY, TEXT, no GPU lock), aggregate tok/s = n / token_s:

| n | complete token | aggregate tok/s | vs N=1 |
|---|----------------|----------------:|-------:|
| 1 | 37.855 ms      | 26.4            | 1.00   |
| 2 | 55.3 ms        | 36.1            | 1.37   |
| 3 | 79.4 ms        | 37.8            | 1.43   |
| 4 | 112.0 ms       | 35.7            | 1.35   |

One child already moves 13.618 GB/token at 401.6 / 411.51 GB/s. Extra
children partition the same DRAM. N=2 takes the leftover host/encoder slack.
N=3 is flat. N=4 is worse **and** swaps (751 MB used, 0.81 GB free, 70.1 GB
wired).

N=8 was **not launched**. 8 × 15.659 GB = 125.3 GB on a 96 GB box.

For short TEXT evals the Metal upload dominates: N=1 / 8 tokens was 5.4–7.3
s; N=4 / 8 tokens via the pool was 42.8 s. Do not cold-start four children
for an 8-token recipe eval.

**Admission default: `safe_n = 3`. Operating point for TEXT search: 2.**

## The pool

```
lab/genesis_pool.py      GenesisPool.spawn / poll / kill
tools/genesis_pool.py    CLI: spawn, poll, kill, measure, e2e, stub-child
```

- `spawn(prompt, budget, hold_gpu_lock=False) -> child_id`  (non-blocking)
- `poll(id) -> running | done(text, wall_ns) | failed(reason)`
- `kill(id)` sends SIGTERM to the process group, then SIGKILL. No orphans.
- Liveness is `kill(0)` + `ps` state + `Popen.poll()`. A status file is
  never read. `test_poll_ignores_stale_status_file` writes `status=done`
  and `result.json` while the child is alive; poll still returns running.
- Stdout/stderr land on disk at spawn. A crashed pool loses no completed
  work.
- `hold_gpu_lock=True` wraps `tools/gpu_lane_lock.sh` (TIMING).
  `hold_gpu_lock=False` is TEXT and must stay that way or the pool
  serializes to one.

### Workload A — gravity recipe (TEXT)

Per-layer, per-tensor assignment. Thousands of cheap-ish evals. Parallelize
up to N=2 (cap 3). Do **not** take the GPU lock.

### Workload B — kernel floor (TIMING)

Launch geometry, fusion, encoder sharing, residency. The generate that
produces a complete-token wall **holds the lock**. The pool's value is
starting candidate N+1's setup (compile / pack / shader, `hold_gpu_lock=
False`) while candidate N holds the lock. Two TIMING children serialize;
that is tested.

## Proofs

- **4 real children concurrently** completed real prompts via the pool
  (seq=128, 8 new tokens, TEXT). Per-child walls 42.72–42.77 s; aggregate
  42.82 s. Output on disk under `workspace/ops/local/genesis-e2e/e2e/`.
  A longer N=4 (1200 tokens, seq=2048) also completed, 136–139 s each,
  111–113 ms complete-token.
- **Kill reclaims memory.** Live child at 15.654 GB phys / 21.436 GB
  wired → SIGTERM → pid gone, RSS gone, wired 6.987 GB, **14.449 GB
  wired returned**.
- **Admission refuses.** `safe_n=1`, second `spawn` raised
  `AdmissionRefused(safe_n=1, alive=1)`. No second process.

`python3 -m pytest -q lab/tests/test_genesis_pool.py` — 17 passed, 1
skipped (live opt-in).

NEXT_BOTTLENECK is unchanged by the pool. The complete token is still
weight addressing:

`weight_addressing, 21293103`
