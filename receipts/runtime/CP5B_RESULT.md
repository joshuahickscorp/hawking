# CP5b — the chunk survives the layer. 2.205x with the real recurrence in the loop.

Raw: `receipts/runtime/CP5B_LAYER_WITH_RECURRENCE.json`, and the paired reference
`receipts/runtime/CP5A_COMPOSITION_PAIRED.json` — a CP5a re-run on the SAME quiet lane, so the
two numbers share a thermal state instead of being compared across one.

CP5a left one structural question open: does a chunk of K positions survive crossing a layer
with the recurrence still stepping one position at a time, or does the recurrence re-serialise
the projections around it? CP0-CP2's WY composition was built on the assumption that it does.

## Arms

    SEQUENTIAL  K command buffers, each in_proj -> recurrence -> gate_up -> down for ONE
                position.  4 command buffers, 16 dispatches.
    BATCHED     ONE command buffer: batched in_proj -> K SERIAL recurrence dispatches ->
                batched gate_up -> batched down.  1 command buffer, 7 dispatches.

The recurrence is `qwen38_gated_delta_decode_vi_simd`, the real kernel, at the real geometry
(48 value heads, key_dim 128, value_dim 128) and the production launch shape
`(kd, heads, vd) / (kd, 1, 1)` with 512 B of threadgroup scratch, read off the runtime's
`encode_gated_delta_fused_ba` rather than guessed. The state buffer is mutated and the update
is order dependent, so both arms reset it from the same seed and step the same K positions in
the same order. Decay sits in (0,1) and beta is small — the regime the gated delta rule
actually runs in; a decay of 1.0 would make the state a plain sum and hide an ordering bug.

## Result

| | cmd buffers | dispatches | seq GPU ns | batched GPU ns | speedup |
|---|---|---|---|---|---|
| CP5a, no recurrence | 4 → 1 | 12 → 3 | 1,897,664 | 799,250 | **2.374x** |
| CP5b, with recurrence | 4 → 1 | 16 → 7 | 2,039,165 | 924,916 | **2.205x** |

`max_rel_err 0.0` on the recurrence AND on in_proj, gate, up and down, in both harnesses. The
K serial dispatches inside the batched command buffer evolve the state exactly as K separate
command buffers do.

**The recurrence does not re-serialise anything.** It costs 141,501 ns across 4 positions in
the sequential arm and 125,666 ns in the batched arm — 35,375 vs 31,416 ns per position, so it
is marginally CHEAPER inside the shared command buffer, which is the per-CB overhead it no
longer pays three times.

**And the degradation is exactly what the serial cost predicts.** Taking CP5a's ratio as the
projection speedup and the measured recurrence as an unbatchable addend:

    (P + R) / (P/2.412 + R) = 2.197        measured 2.205

Within 0.4%. The recurrence behaves as a fixed serial cost added to both arms and nothing more.

## Two things this corrects

**CP5a's own number moved.** It measured 2.458x earlier and 2.374x here on the same code. That
is the thermal state, not the code, and it is why the reference was re-run on the same lane
rather than quoted from the earlier receipt. Cross-run absolute comparisons on this machine are
not safe; CP3b established that and it still holds.

**The recurrence is cheaper here than `CP5_DECOMPOSITION.md` assumed.** That receipt used
55,625 ns/position from `_DELTANET_ORGAN_raw.json`, which measured
`qwen38_gated_delta_decode_vi_simd_ba` — the FUSED variant, which also derives decay and beta
from `A_log` and `dt_bias`. This harness runs the simpler `_vi_simd` with those precomputed, at
35,375 ns. **The production path uses the fused one**, so 55,625 remains the number for the
step decomposition and the 9.8% share stands. The 35,375 figure describes this harness only.

## What is still not proven

* **One layer, not a sequence.** Nothing here shows K positions crossing from layer L to L+1.
  The structural argument is that the recurrence emits K outputs before the next layer's
  projections begin, and this measurement is consistent with it, but a two-layer harness is
  what would test it.
* **Not the real weight bytes**, not a residency test, and **not a prompt wall**. All three
  kernel families still have **zero call sites** in `qwen38_hybrid_decode.rs`.
* Norms, causal conv, the gated RMSNorm, out_proj and attention are all absent.

## Where the integration actually stands

The physics question is answered as far as isolated harnesses can answer it: batching wins
2.2-2.5x on every organ measured, composes in one command buffer, and survives a serial
recurrence in the middle of that command buffer. What remains is not a measurement but an
integration — teaching `qwen38_hybrid_decode.rs` to step a chunk instead of a token, which is
a change to the per-token loop at line 7566 and every organ encoder it calls. That is runtime
mutation, which under the standing single-writer rule belongs to HCLI, not to a harness.
