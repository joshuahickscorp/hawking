# CP6f -- the chunked MLP is CORRECT on real weights

2026-09-04 · frontier G019 · resident loaded, `NOETIC_PARENT_A`

CP6a/b/c measured the chunked organ on the real session and were **timing
only**; correctness rested on CP4's bit-identical result for the gate_up kernel
in ISOLATION. That is a weaker claim than it sounds -- an organ can be correct
alone and wrong once composed, because composition is where layout assumptions
meet. This closes it.

`verify_chunked_dense_mlp(layer, r, chunk)` runs the real MLP -- head norm,
fused gate_up+swiglu, down_proj, residual tail -- for K positions in one pass on
real catalog weights, against K runs of the production `encode_dense_mlp`.

## Result

```
  layer   r  k    max_abs_resid    max_abs_norm   disp seq -> chunked   bar
      0   1  1       3.5763e-7        0.0000e0     4 -> 4     floor  PASS
      0   4  2       3.8743e-7        0.0000e0     8 -> 4   7.15e-7  PASS
      0   4  4       4.7684e-7        0.0000e0    16 -> 4   7.15e-7  PASS
     31   1  1       1.0729e-6        0.0000e0     4 -> 4     floor  PASS
     31   4  2       1.0729e-6        0.0000e0     8 -> 4   2.15e-6  PASS
     31   4  4       1.0729e-6        0.0000e0    16 -> 4   2.15e-6  PASS
     63   1  1       2.4438e-6        0.0000e0     4 -> 4     floor  PASS
     63   4  2       2.4438e-6        0.0000e0     8 -> 4   4.89e-6  PASS
     63   4  4       2.4438e-6        0.0000e0    16 -> 4   4.89e-6  PASS
```

Three things this says.

**Widening K costs nothing.** At layers 31 and 63 the error is *identical*
across r1k1, r4k2 and r4k4 -- to the last digit. At layer 0 it moves 3.58e-7 to
4.77e-7, about one ulp at that magnitude. All the divergence from production is
the matvec-to-matmul family change; batching K adds none of it.

**The head norm is bit-identical in composition.** `0.0000e0` in every arm
confirms CP6e's interleaved kernel on real weights inside the real encoder, not
just against a fixture.

**Dispatches 16 to 4 at K=4** -- constant in K, which is the mechanism.

## Two corrections this run forced

**The r1k1 bar was wrong.** It was written as "must be exactly zero, because at
one position the layouts are byte-identical". Production runs the **matvec**
family; the chunked path at K=1 runs `..._matmul_..._r1k1...`. Different
kernels, different rows-per-thread accumulation. r1k1 was never going to be
exact and the first run correctly reported FAIL against a bar that could not be
met.

The replacement is stronger than the original intent. r1k1 is not gated against
a constant, it *becomes* the bar: it measures the family change alone, and a
wider arm must not exceed it. That asks the question the chunked path has to
answer -- does batching K cost accuracy? -- instead of asking whether a
hand-picked epsilon was generous enough.

**`fuse_add_rmsnorm` is not what `apply_fusion` sets.** It defaults to the
**fast profile** flag (`Err(_) => (fast, false)`), so the ordinary default path
is the UNFUSED tail. `CP6E_RESULT.md` called the fused kernel "the one the
production path actually runs" -- that is conditional on the fast profile, not
unconditional. Both tails are load-bearing and both are now wired.

## The rule the norms actually violate

The unfused tail needed **no new kernel**. `qwen_next_add_residual` takes an
element count and runs unchanged on the interleaved buffer.

So the correction in `CP6E_RESULT.md` was directionally right but too broad. The
precise rule:

> An elementwise op with **no reduction** and **no position-shared operand** is
> layout-agnostic -- a wider launch is genuinely the whole change. A **reduction**
> (the norms) or a **shared operand** (the norm weight, which is per-hidden-dim
> and must NOT be strided) needs its own interleaved variant.

Being on the chunked path is not what forces a new kernel; reducing or sharing is.

## Mutation check -- and what it caught about the bar

Dropping the head norm from the chunked path:

```
      0   4  2        4.5879e1        1.9932e0     8 -> 3   7.83e1  FAIL
     31   4  4        2.5118e0        3.3862e0    16 -> 3   4.93e0  FAIL
```

All 9 arms FAIL. But look at which check bit: `4.5879e1` is **under** its own
bar of `7.83e1`. **The floor-relative gate alone would have passed a
catastrophically broken chunked path** -- because the mutation moved the floor
too, and a relative bar is blind to a defect that corrupts its own reference.

What caught it was the absolute `max_abs_norm == 0.0` gate on the output that is
bit-identical by construction. Keeping one exact check beside the relative one
is what makes this suite able to fail.

Marker `MUTATION_CP6F` grepped to 0 after restore; rebuilt, re-run, all arms
PASS.

## Standing

The chunked MLP is correct and dispatch-cheap on real weights. It is not yet in
`encode_layers` -- the mixer half still needs per-position addressing, and until
that lands the prompt wall cannot be re-measured. That, not correctness, is what
G019 still owes.
