# CP5 — what integration actually requires, decomposed from measurement

CP3 (2.474x, uniform-q4 in_proj), CP4 (2.49x, affine gate_up) and CP4b (2.27x, affine
down_proj) are all ORGAN measurements at one layer in isolation, and all three kernel
families still have **zero call sites** in `qwen38_hybrid_decode.rs`. The open question is
whether any of it becomes a prompt wall.

The assumption going in was that integration is gated on the CP0-CP2 WY chunk composition,
because the DeltaNet recurrence is the sequential part. **Measurement says that assumption is
wrong about the size of the problem.**

## The step, decomposed

Sources: `receipts/headless/_ORGAN_BANDWIDTH_raw.json` (`isolated_organs`, 7 reps, noop
control 125 ns) for organ shares, and `receipts/headless/_DELTANET_ORGAN_raw.json`
(`parity.*.baseline_gpu_ns`) for the pure recurrent state kernel — 55,625 ns for one
dispatch, x 48 DeltaNet layers.

| class | share of step | status |
|---|---|---|
| mlp_gate_up | 35.9% | position-independent, **2.49x measured** (CP4) |
| mlp_down | 20.9% | position-independent, **2.27x measured** (CP4b) |
| deltanet minus recurrence | 14.3% | position-independent; in_proj is the CP3 organ itself |
| q4_remainder | 7.7% | position-independent, uniform q4, CP3 family applies |
| **batchable subtotal** | **78.7%** | **no recurrence work required** |
| deltanet recurrence | **9.8%** | strictly sequential; this is what WY is for |
| gqa_attention | 6.5% | needs chunked attention over a growing KV |
| lm_head + sampling | 5.0% | prefill needs these for the LAST position only |

The pure recurrence is **9.8% of the step**, not the majority. It is 40.8% of the deltanet
organ, and the other 59.2% of that organ — in_proj, causal conv, norms, out_proj — is
position-independent like the MLP.

## What that reorders

A partial integration that batches only the position-independent organs and leaves the
recurrence stepping one position at a time would reach, on the measured per-organ speedups:

    batchable 78.7% at 2.27x  ->  step GPU ns x 0.560  =  1.79x
    batchable 78.7% at 2.49x  ->  step GPU ns x 0.529  =  1.89x

So **the WY composition is not the blocker for most of the win.** It is worth at most the
last ~10%, and it is the hardest piece. CP5 should therefore be: batch the position-independent
organs across a chunk of positions, keep the recurrence sequential, and measure the prompt wall.
That is a far smaller change than a full chunked-prefill port and it is where the evidence points.

## What this is NOT

**This is arithmetic over organ measurements, not a prompt-wall measurement.** This mission has
already had to retract exactly one projection of this shape — the 4.281x "vs serial-K1" ratio
that CP3b killed — and the discipline that caught it applies here too. Specifically unproven:

* that a chunk of K positions can be driven through a layer without the recurrence forcing a
  per-position boundary that re-serialises the projections anyway;
* that the interleave transform (`input[col*K + k]`) stays cheap when it feeds a real layer
  rather than a benchmark buffer — CP3 charged it at 15.9% of the GPU time it saved;
* that per-organ speedups compose. Organs measured alone do not share a command buffer, and
  production runs one CB per token;
* that residency holds. Batching K positions multiplies every activation and temporary by K.

The number to beat is the complete prompt wall against the retained sequential baseline, and
nothing here has measured it.

## Next

1. Batch ONE layer's position-independent organs over K positions on the real path, recurrence
   still sequential, and measure that layer's wall against the sequential baseline. This is the
   first measurement that can falsify the composition assumption.
2. Then multi-layer, then the full prompt path (CP6), then the complete WorkUnit wall (CP7).
3. The WY composition stays on the frontier but is now correctly ranked: hardest piece,
   worth ~10%, and it does not block the other 79%.
