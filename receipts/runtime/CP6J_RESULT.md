# CP6j -- the chunk curve, and why K=8 lost

2026-09-04 · frontier after G019 · 128 prompt tokens, all 64 layers, full resident

CP6i capped `chunk` at 4 because the fused swiglu family was only instantiated
to (4,4). That was an accident of the shader, not a physical limit, so the cells
were added and the curve mapped. Every row is capability-checked; none of these
ratios is reported without its sampled token matching.

```
   r   k   R*K   speedup   conservative   sampled     verdict
   4   2     8    1.133x        1.065x    95726 ==    PASS
   2   4     8    1.314x        1.303x    95726 ==    PASS
   4   4    16    1.348x        1.339x    95726 ==    PASS
   4   4    16    1.404x        1.372x    95726 ==    PASS
   2   8    16    1.331x        1.322x    95726 ==    PASS
   4   8    32    0.780x        0.771x    95726 ==    FAIL (slower)
```

## What binds it is R*K, not K

`r4k8` is **0.780x -- a loss**. The obvious reading is that chunk 8 is too wide.
It is not. `r2k8` is the discriminator: same K, same chunk geometry, same number
of positions in flight, **half the accumulators**. It comes back at **1.331x**.

So chunk width is not what breaks at 8. The binding quantity is the product:

```
  R*K =  8   1.13 - 1.31x    under-fed; too few FMAs per unpacked weight
  R*K = 16   1.33 - 1.40x    the plateau
  R*K = 32   0.78x           collapses
```

Each thread holds `2*R*K` live accumulators (gate and up). At r4k8 that is 64,
and the fall from 1.33x to 0.78x across one doubling is the shape of a
register-spill cliff rather than a gradual efficiency loss.

This confirms the first of the three candidates CP6c named -- register pressure,
reduction cost, occupancy -- and separates it from the other two, which r2k8 and
r4k4 hold roughly equal while R*K stays at 16.

## Consequence

**`R*K <= 16` is the operating rule for this kernel family on this GPU**, and
within it R and K trade freely: r4k4 and r2k8 are within 5% of each other. So a
wider chunk is available when something else wants it -- more positions per
command buffer, fewer glue dispatches -- as long as R comes down to match.

The best measured configuration remains **r4k4 at 1.35-1.40x**, with r2k8 a
statistically indistinguishable alternative at twice the chunk width.

## Reopen conditions

- A kernel that stages accumulators through threadgroup memory instead of
  registers would move the cliff, and R*K=32 is where it would pay.
- CP6h's discarded LM head was 4.3% of the sequential token wall. That wall is
  now ~1.4x shorter while the head is unchanged, so its share has risen; it is
  still small, but the arithmetic that set it aside has moved.
