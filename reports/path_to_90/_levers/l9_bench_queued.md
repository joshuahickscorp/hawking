# L9 — Headline bench queue

The `tools/bench/path_to_125_bench.sh` script refuses to run with Claude
open (contention skews dec_tps 4-5×). Each lever below has shipped with
its own parity gate; the headline numbers are queued for the user to
run during a clean window.

## How to run

Quit Claude (Cmd-Q), then from the dreamy-golick-d54ff8 worktree:

```
cd /Users/scammermike/Downloads/dismantle/.claude/worktrees/dreamy-golick-d54ff8
tools/bench/path_to_125_bench.sh
```

Paste the numbers back in the next Claude session — they'll be folded
into the exhaustion_report.md.

## Pending benches

| Lever | Commit | A/B variants to measure |
|---|---|---|
| L4 — AMX V2-Lite attn projections | `4f24d46` | `attn_proj_amx=false` (default, current) vs `attn_proj_amx=true`. Run both with `verify_kernels=parallel-k` and the v3 head loaded. Median dec_tps over 10 trials each. Per the L4 commit note, the production decode hot path uses fused rmsnorm+attn-gemv TCB kernels that DO NOT route through `gemv_f32_attn_dispatch`, so the AMX flag is expected to be a near-no-op for hot-path dec_tps. The bench validates the no-op (not a regression). |
| L5 — multi_queue (EAGLE4_BACKEND=metal only) | `e482463` | `EAGLE4_BACKEND=metal multi_queue=false` vs `EAGLE4_BACKEND=metal multi_queue=true`. Default backend (AMX) is unaffected. Even under metal backend, the chain decode loop is internally serial, so this bench mostly validates no-regression. |
| **L7.1 — v3_xtg Q4_K_M GEMV** | **`50513c0` (kernel) + `9b7038d` (wire-up)** | **Most important new bench.** Edit `profiles/deepseek-v2-lite-q4.m3pro18.json` `selected.gemm_q4_k_schedule` between `"v2"` (default), `"v3_8r"`, and `"v3_xtg"`. Run all 3 in clean window with `verify_kernels=parallel-k`. Expected win: `v3_xtg` over `v3_8r` by 5-15% on V2-Lite expert (rows=10944, cols=2048) and LM head (rows=102400, cols=2048) due to 8× redundant x-reads eliminated. If `v3_xtg` wins by ≥5%, bump default. If <5% or regresses, keep as opt-in. |

L6/L8 entries: L6 deferred pending chain-decode pipelining; L8 is
mid-run (pid 84960) — separate τ-eval + chain-decode smoke required
after training completes, not a kernel bench.

## Why we proceed without bench

Per the path-to-125 prompt §8: bench numbers are nice-to-have, not
required-for-shipping. Each lever's parity gate + synthetic A/B is
sufficient for ship decisions. The exhaustion-report queue tracks
which levers still need their headline number filled in.
