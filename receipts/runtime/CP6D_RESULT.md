# CP6d — the prompt wall, measured. The baseline G019 requires beating.

Raw: `receipts/runtime/CP6D_PROMPT_WALL.json`
Harness: `crates/hawking-core/examples/ascension_qwen38_cp6d_prompt_wall_baseline.rs`

Every prior receipt in this campaign divides one organ by another. G019's acceptance is a
complete prompt wall against the retained sequential baseline, and that baseline had never been
measured here. One new token per run, so decode cannot hide the quantity under test. Zero loaded
residents, under `gpu_lane_lock.sh`.

| prompt tokens | prefill wall | ns/position | fresh-compute tok/s | GPU % of wall | dispatches/pos |
|---|---|---|---|---|---|
| 128 | 3,071.5 ms | 23,996,096 | **41.67** | 94.7 | 708 |
| 256 | 6,110.1 ms | 23,867,590 | **41.90** | 96.4 | 708 |
| 512 | 12,621.5 ms | 24,651,395 | **40.57** | 96.5 | 708 |
| 1024 | 26,973.6 ms | 26,341,432 | **37.96** | 96.7 | 708 |

**Standing fresh-compute prompt throughput on this build: 38–42 tok/s.** The directive's
standing record says ~36, so the current build is slightly ahead of it, and the milestone ladder
(50, 75, 100, 150, 200+) is unchanged.

## Prefill is near-linear here, which corrects a documented premise

Per-position cost across an **8x** length range moves **9.8%**: 23.996 → 26.341 ms. That is
near-linear, and the rise is consistent with the GQA KV cache growing — attention is 6.5% of a
step and is the only term that scales with position.

`hcli/prefill_profile.py` opens by documenting prefill as superlinear, from 2500 tokens in
170.7 s and 3099 in 385.4 s, and builds a whole bucketing analysis on that. Those figures are
**68.3 and 124.4 ms/position**, against 24.0–26.3 here — 2.6x to 4.7x slower per position. That
is a different build or a different regime, not this one.

Stated carefully, because the ranges do not overlap: **at 128–1024 tokens on this build prefill
is near-linear.** The superlinear observation was made at 2500–3099 tokens and at several times
the per-position cost. Whether a crossover exists between 1024 and 2500 is **unmeasured**, and
that is a cheap experiment for whoever needs it. What is refuted is only the assumption that
superlinearity is a standing property of this prefill path.

## The host is not the lever

GPU accounts for **94.7–96.7%** of the prefill wall. Host work outside the GPU is 3–5%.
Directive XIX asks for CPU/GPU overlap; on this path it cannot be worth more than 5%, and that
is a ceiling, not an estimate. Dispatches are a flat **708 per position** at every length.

## What batching would give — arithmetic, not measurement

Substituting measured organ shares at measured batched speedups:

    gate_up alone (35.9% of step, 1.903x)          x0.8296
      128 tok    3,071.5 -> 2,548.3 ms     41.67 -> 50.23 tok/s
      1024 tok  26,973.6 -> 22,378.6 ms    37.96 -> 45.75 tok/s

    all position-independent organs (78.7%, ~1.9x)  x0.6272
      128 tok    3,071.5 -> 1,926.5 ms     41.67 -> 66.44 tok/s
      1024 tok  26,973.6 -> 16,918.1 ms    37.96 -> 60.52 tok/s

**This is arithmetic over a measured share and a measured speedup. It is not a measured prompt
wall.** This campaign has already retracted two projections of exactly this shape — CP3b killed
a 4.28x, and CP6b killed my own prediction that fusion would lift 1.87x to 2.4x. The value of
stating it is that CP6 stage 1 can now falsify a specific number rather than a direction.

Read against the ladder: batching gate_up alone would put the 128-token case just over the
**50 tok/s** milestone and leave 1024 short of it. Batching every position-independent organ
would clear 50 at both and approach 75 at short prompts. Neither is claimed.

## What G019 still needs

The measured baseline now exists. What does not exist is a chunked prefill that produces a
second row of this table. That is CP6 stage 1 — the K-wide workspace and the 7 mechanically
K-parallel kernels — and until it runs, every number above the line in this receipt is a
baseline and every number below it is arithmetic.
