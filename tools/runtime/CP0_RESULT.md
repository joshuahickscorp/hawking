# R3 CP0/CP1/CP2 — chunked prefill for the gated delta rule

## The recurrence, read off the kernel

`shaders/qwen_next.metal::qwen_next_gated_delta_decode_single`, per head:

```
S <- d*S ;  kv = S^T k ;  delta = (v - kv)*b ;  S <- S + k (x) delta ;  out = S^T q
```

which is affine in S:

```
S_t = A_t S_{t-1} + B_t
A_t = d_t (I - b_t k_t k_t^T)     K x K, scaled rank-1 update of I
B_t = b_t k_t v_t^T               K x V, rank one
```

Affine maps compose, so T positions collapse to one `(A, B)` and the K x V
state is touched once per chunk instead of T times.

## CP1/CP2 — proven numerically, not argued

`tools/runtime/chunked_delta_reference.py`, numpy, under a second:

| check | result |
|---|---|
| token-step vs affine composition, T = 1,2,3,8,32,64,128 | max rel 3.05e-15 |
| `prod(A_t)` vs its WY form `I - K^T W` | max rel 1.11e-14 |
| two unequal chunks composed at the boundary, splits 1/7/16/63 | max rel 3.13e-15 |

Machine precision. The composition closes exactly.

## Operation classification

| operation | class |
|---|---|
| RMSNorm, q/k/v/ba projections, output projection, MLP gate/up/down | BATCHABLE (GEMM over T) |
| causal conv, kernel 4 | BATCHABLE with a 3-position halo |
| GQA layers (16 of 64) | BATCHABLE, ordinary blocked attention |
| `prod(A_t)` via WY | STATE_COMPOSITION, short recurrence over T rows of length K |
| the single state apply `A S + B` | STATE_COMPOSITION, once per chunk |
| nothing found | STRICTLY_SEQUENTIAL |

## Where the win comes from — and where it does NOT

**Not FLOPs.** Chunking is FLOP-neutral to slightly worse: measured ratio 0.44x
at T=16, 1.33x at T=128, because the WY build is O(T^2 K). Anyone selling this
as an arithmetic saving is wrong.

**Bandwidth.** The model is DENSE -- `qwen38_geometry.rs` refuses MoE keys
outright -- so every position re-reads the full weight set.

CORRECTION, 2026-09-04. An earlier version of this file estimated 20.5 GB of
weights from geometry at one byte per parameter and derived "75% of peak" from
it. The resident REPORTS its own figure on the ready banner:

    resident_weight_bytes = 10,554,259,456   (10.55 GB)

Half the estimate. The reported number is authority; the estimate was mine.
Recomputed from a clean R1 run -- 586 positions of cold prefill in 15.9 s, so
27.1 ms per position:

    10.55 GB / 27.1 ms  =  389 GB/s  =  50% of the measured 778.8 GB/s peak

Across the earlier campaign aggregate, 17,048 positions imply 180 TB, needing
231 s at peak against 595 s observed -- 39% of peak.

50% of peak bandwidth on a GEMV-dominated loop is still a bandwidth-limited
regime: a dense GEMV is close to a pure streaming read and rarely clears 60-70%.
But it is NOT the 75% first published, and the honest headline is weaker: there
is roughly a 2x bandwidth headroom here, not a 4x one. A GEMM over T positions
still reads those weights once instead of T times, which is the mechanism; the
size of the prize is now a measured 2x on the bandwidth term rather than an
estimated 4x.

## Projected

Prefill is 75% of accepted-unit model wall (595 s of 799 s), at 28.7 prompt tok/s.

| prefill speedup | prefill wall | model wall | prompt tok/s |
|---|---|---|---|
| 1x (today) | 595 s | 799 s | 28.7 |
| 2x | 298 s | 502 s | 57.2 |
| 4x | 149 s | 353 s | 114.6 |
| 8x | 74 s | 278 s | 229.2 |

The 2x row is the one the corrected bandwidth arithmetic supports directly. The
4x and 8x rows require the chunked path to also win back dispatch and launch
overhead, which CP3 measures rather than assumes.

## Next rungs

CP3 one organ one layer physical, CP4 measured prompt throughput against the
sequential baseline, CP5 multi-layer, CP6 full resident prompt path, CP7 complete
WorkUnit wall. Acceptance stays capability-equivalence AND lower complete prompt
wall -- not "chunked code runs".
