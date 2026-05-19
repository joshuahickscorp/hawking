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

L5/L6/L7/L8 entries will be appended as each lever lands.

## Why we proceed without bench

Per the path-to-125 prompt §8: bench numbers are nice-to-have, not
required-for-shipping. Each lever's parity gate + synthetic A/B is
sufficient for ship decisions. The exhaustion-report queue tracks
which levers still need their headline number filled in.
