# CP6c — K is the only variable. R does nothing, and two standing priors are wrong.

Raw: `receipts/runtime/CP6C_WHAT_BINDS.json`
Harness: `crates/hawking-core/examples/ascension_qwen38_cp6c_what_binds_the_chunk.rs`

CP6b ruled out unpack cost by experiment. Three candidates remained for what holds the chunked
gate_up at 30% of roof: register pressure, reduction cost, and occupancy. Real catalog weights,
64 layers, alternated against the production baseline, first rep discarded.

| r | k | TGs | acc | sweep ns | ns/position | FMA per weight | GB/s | % roof |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 8704 | 2 | 6,875,666 | 6,875,666 | 2 | 518.5 | 66.6 |
| 2 | 2 | 4352 | 8 | 11,757,416 | 5,878,708 | 4 | 303.2 | 38.9 |
| 4 | 2 | 2176 | 16 | 11,528,708 | 5,764,354 | 4 | 309.2 | 39.7 |
| 2 | 4 | 4352 | 16 | 15,267,916 | 3,816,979 | 8 | 233.5 | 30.0 |
| 4 | 4 | 2176 | 32 | 15,321,416 | 3,830,354 | 8 | 232.7 | 29.9 |

The control holds first: **r1k1 lands at 1.060x the production baseline**, which is where a
degenerate one-position arm must land. Had it not, no cell below would be readable.

## R changes nothing

    K=2:  r2 5,878,708  vs  r4 5,764,354   ->  1.020x    TGs 4352 vs 2176
    K=4:  r2 3,816,979  vs  r4 3,830,354   ->  0.997x    TGs 4352 vs 2176, acc 16 vs 32

Doubling R halves the threadgroup count and doubles the accumulators, and the time does not
move. Both remaining candidates die here:

* **Occupancy is not the constraint.** 2176 threadgroups against 60 cores performs the same as
  4352, and 8704 at r1k1 is no better per FMA.
* **Register pressure is not the constraint.** 32 live accumulators performs the same as 16.

**This refutes CP4b.** That receipt concluded "the governing variable is grid width against core
count, not registers", from a harness where gate_up (17408 rows) and down_proj (5120 rows) had
different knees. On the real organ, at real weights, across 64 layers, grid width does nothing
across a 4x range. CP4b's conclusion was drawn from two organs of different SHAPES and attributed
to grid width; it does not survive a direct test at fixed K.

## K is the only variable, and the mechanism is FMA count

    K=1   2 FMAs per weight loaded   sweep  6,875,666 ns   1.060x
    K=2   4 FMAs per weight loaded   sweep 11,528,708 ns   1.264x
    K=4   8 FMAs per weight loaded   sweep 15,321,416 ns   1.903x

The weight traffic is identical in every row — one sweep of 3.565 GB. What changes is the
arithmetic riding on it: `2*K` fused multiply-adds per weight loaded. Sweep time grows with K but
**sublinearly** — 4x the FMAs costs 2.23x the time — and that gap is exactly the batching win.

The transition is visible in the bandwidth column. At K=1 the kernel runs at 518.5 GB/s, 67% of
the 778.8 GB/s roof: it is bandwidth-bound, and batching has room. At K=4 it is at 232.7 GB/s,
30%: the traffic saving has been banked and **FMA throughput is now what binds it**.

## What this means for the next lever

Not more batching: K=4 already sits at 8 FMAs per weight and the returns are compressing.
Not fewer registers, not better occupancy — both measured to do nothing here.

The remaining lever on this organ is **arithmetic throughput per weight**: fewer FMAs for the
same result, or the same FMAs issued better. Apple's simdgroup matrix instructions are the
obvious candidate, since the inner loop is now a small dense matrix product per weight
(`R` rows x `K` positions) being executed as scalar FMAs.

That is a hypothesis with a clear test and it is NOT claimed here. What is claimed is narrower
and measured: **K is the only variable that moves this kernel, and at K=4 it is FMA-bound, not
bandwidth-, register-, or occupancy-bound.**

## Standing numbers after CP6c

    gate_up, real weights, 64 layers, against the production path it replaces
      K=4   1.903x   (r2k4 and r4k4 within 0.4% of each other)

Consistent with CP6a's 1.890x and CP6b's 1.890x on separate lanes — three independent runs
agreeing to 0.7%.
