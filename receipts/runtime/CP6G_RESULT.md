# CP6g -- the HYBRID layer wins, with the glue paid

> **QUALIFIED 2026-09-04 by `CP6I_RESULT.md`.** These numbers were measured on a
> **racing encoder**. Arm B never called `begin_serial_group`, so within a layer
> position k's gather overwrote `workspace.hidden` while k-1's mixer was still
> reading it. CP6g measured only time and could not see it. Serialising the chunk
> cost nothing measurable (1.346x to 1.349x on the prompt wall), so the ratio here
> is not believed to be inflated by the race -- but it was not measured on a
> correct encoder, and CP6i's prompt-wall figures, which were, supersede it.

2026-09-04 · frontier G019 · resident loaded, `NOETIC_PARENT_A`, quiet lane

## The idea

The mixer is genuinely per-position: DeltaNet carries a recurrent state and GQA
appends KV, so k+1 depends on k. Batching it means threading per-position
addressing through many encoders -- wide, mechanical, and the standing blocker
on G019.

The MLP has no such dependency and it is the larger half of the step. So the
question is whether a chunked prefill has to wait for the mixer at all: run the
mixer K times exactly as it runs today, scatter each result into slot k of a
K-wide buffer, run the MLP **once**, gather back.

The glue is paid inside the measurement -- 2K extra dispatches (a scatter and a
gather per position) against K-1 MLPs saved. If the trade were bad this would
report a loss.

## Result -- 64 layers, three independent runs

```
  layers  r  k   seq ns/pos  hyb ns/pos   speedup  [conservative-generous]   seq jitter
     64   4  2       396842      363969    1.090x  [0.981-1.244]              11.1%   not separated
     64   4  2       377329      378504    0.997x  [0.940-1.183]              10.9%   not separated
     64   4  2       396951      361218    1.099x  [0.991-1.161]               8.7%   not separated

     64   4  4       385979      304104    1.269x  [1.218-1.309]               2.4%   SEPARATED
     64   4  4       389530      296113    1.315x  [1.257-1.354]               3.1%   SEPARATED
     64   4  4       388630      291562    1.333x  [1.288-1.394]               2.2%   SEPARATED
```

**K=4 wins, 1.27-1.33x, reproducible within 5%, conservative bound above 1.0 in
all three runs.** Conservative = slowest hybrid over fastest sequential; if that
straddled 1.0 the arms would not be separated and nothing would be claimed.

**K=2 does not win and is reported as not winning.** Its dispatch count is
*identical* -- 1408 to 1408 -- because 2K glue dispatches exactly cancel the MLP
launches saved. K=4 is 2816 to 2560.

## The instrument had to be fixed first, and that is the transferable part

The first version measured **one layer** and produced this:

```
      0   4  4   ...  1.745x  [0.768-2.161]  seq jitter  47.7%
      3   4  4   ...  1.817x  [0.595-2.169]  seq jitter  53.5%
      1   4  2   ...  1.067x  [0.504-2.674]  seq jitter 130.0%
```

A 1.817x that would have been very easy to report. Nothing was separated -- the
conservative bounds sit at 0.5-0.77 -- and the baseline moved 36% between two
arms of the same measurement.

Before blaming noise the series was checked for **monotone drift**, because both
arms append to the KV cache and advance the recurrent state every rep, and
growing work would look like jitter in a median. Second-half-over-first came
back **0.84-1.08**, and the raw series bounces (`939, 755, 1059, 930, 810, 771,
1094, 702 us`) rather than climbing. So: genuine noise, not accumulation.

The cause is scale. A single layer is a ~400 us command buffer, below this
instrument's floor. CP6a/b/c reached 0.7% reproducibility by putting all 64
layers in one command buffer, and doing the same here took jitter from 47-130%
to **2.2-3.1%** at K=4.

Two things worth keeping:

1. **A median over a drifting quantity is not a measurement, and a median over a
   noisy one is not evidence without its spread.** Reporting min/median/max and
   a conservative bound is what turned a spurious 1.817x into an honest
   "not separated".
2. **Rule out systematic drift before calling something noise.** The
   discriminator cost one ratio and it was the difference between "the
   instrument is too small" and "the KV cache is growing under us".

## What this does NOT establish

This is the **layer wall**, not the prompt wall. G019's acceptance is a lower
complete prompt wall against the CP6d baseline, and that requires the chunked
path to be reachable from `encode_layers` -- it is not; `encode_layers` still
calls only the per-position path.

A projection from 1.3x on the layer wall to the CP6d row would put 41.67 tok/s
near 54 tok/s, but that is an **ESTIMATE, not a measurement**, and it assumes
embedding, final norm and LM head are negligible, which has not been measured
here.

Correctness of the chunked MLP itself is CP6f: error identical across r1k1,
r4k2 and r4k4, so widening K costs nothing.
