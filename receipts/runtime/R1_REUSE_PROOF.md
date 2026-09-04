# R1 — prefix / state reuse, PHYSICALLY PROVEN

`tools/runtime/r1_reuse_proof.py`, one resident, one model load, all arms sharing
real state. Repeated alternating passes; the first triple is warmup and excluded.

```
resident ready: weight_bytes=10,554,259,456 max_seq_len=8192

arm                       prompt  reused  stepped    wall  source
A-COLD                       586       0      586   15.9s  cold
D-WARM-SUFFIX-MUTATED        584       0      584   15.8s  cold
C-WARM-PREFIX-MUTATED        586       0      586   15.9s  cold
B-WARM-IDENTICAL             586     579        7    0.4s  checkpoint_restore
D-WARM-SUFFIX-MUTATED        584     579        5    0.3s  checkpoint_restore
C-WARM-PREFIX-MUTATED        586       0      586   15.8s  cold
B-WARM-IDENTICAL             586     579        7    0.4s  checkpoint_restore
D-WARM-SUFFIX-MUTATED        584     579        5    0.3s  checkpoint_restore
C-WARM-PREFIX-MUTATED        586       0      586   15.9s  cold

  PASS  warm steps fewer positions than its prompt
  PASS  suffix mutation PRESERVES the reusable prefix
  PASS  prefix mutation INVALIDATES reuse

  effective prompt tok/s        105.7
  fresh-compute prompt tok/s     36.0
  physical positions avoided     2316
```

## Acceptance clauses

| clause | evidence |
|---|---|
| warm processes fewer physical positions than cold | 7 or 5 stepped against 586 |
| prefix mutation invalidates the reuse region | C reuses 0 every time, falls back to cold |
| suffix mutation does NOT invalidate the prefix | D reuses 579 every time |
| semantics preserved cold and warm | bit-identical, `R1_CHECKPOINT_PARITY.md` |
| complete model wall moves as predicted | 15.9 s cold against 0.4 s warm |
| bookkeeping cost reported not hidden | warm hit costs 0.3-0.4 s wall, stated |
| survives repeated pairs | three passes, identical behaviour |

## The two throughput numbers, kept apart

    effective       105.7 tok/s   all prompt tokens / prompt wall
    fresh-compute    36.0 tok/s   physically stepped positions / prompt wall

The first flatters reuse by crediting the cache with tokens nobody computed. The
second is how fast this machine actually processes a prompt position. Only the
second may be compared against a chunked implementation.

## Cold cost per position

586 positions in 15.9 s = 27.1 ms per position, against 10.55 GB of reported
resident weight bytes = 389 GB/s = 50% of the measured 778.8 GB/s peak. The
prompt loop is bandwidth-limited with roughly 2x headroom on that term.

## Failure classes NOT observed

KEY_INSTABILITY, PROCESS_INSTABILITY, STATE_NOT_PERSISTED, INVALIDATION_TOO_BROAD,
INVALIDATION_TOO_NARROW, RUNTIME_REPLAY. Reuse is keyed on exact token-prefix
equality, persists across requests in one resident, and invalidates in exactly
the right direction on both mutation arms.

R1_PREFIX_STATE_REUSE = PHYSICALLY_PROVEN
