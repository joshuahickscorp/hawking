# CP5c — a chunk crosses a layer boundary, at no cost and with no transpose

Raw: `receipts/runtime/CP5C_TWO_LAYERS.json`, paired reference
`receipts/runtime/CP5B_PAIRED_REFERENCE.json` (a CP5b re-run on the SAME quiet lane).

CP5b answered one layer. The last structural question before integration is whether a chunk of
K positions survives crossing from layer L to L+1 — and there is a specific claim underneath it
that decides the integration cost model:

> the multi-position OUTPUT layout is `out[row * K + k]`
> the multi-position INPUT layout is `input[col * K + k]`

`down_proj` emits 5120 rows, which is the hidden size, and the next layer's `in_proj` contracts
over 5120 columns. So layer L's output should ALREADY BE layer L+1's input layout, and the
interleave is paid once per chunk rather than once per layer. If false, integration pays a
transpose 64 times and CP3's 15.9% transform charge recurs at every layer.

## Result

| | cmd buffers | dispatches | seq GPU ns | batched GPU ns | speedup |
|---|---|---|---|---|---|
| CP5b, one layer | 4 → 1 | 16 → 7 | 1,382,998 | 648,499 | **2.135x** |
| CP5c, two layers | 4 → 1 | 32 → 14 | 2,753,082 | 1,290,624 | **2.131x** |

**The ratio holds. Drift −0.2%.**

The second layer adds 1,370,084 ns to the sequential arm and 642,125 ns to the batched arm — a
ratio of 2.134, indistinguishable from layer 1's. Layer 2 costs exactly what layer 1 costs, in
both arms, with **no crossing penalty**.

Layer 2 is bound DIRECTLY to layer 1's `down` output: no copy, no re-interleave. It agrees with
the sequential arm to **2.095e-7 of tensor magnitude** — f32 epsilon is 1.19e-7, so one to two
ulps. Layer 1 and both recurrences are bit-identical.

**A wrong layout would make layer 2 read different values entirely and disagree by order 1, not
by two ulps.** Agreement at rounding is the proof, not a hope.

## Three harness defects, found and fixed — none of them findings about the runtime

1. **Both layers shared one recurrent state buffer** → `max_rel 2.342e2`. The sequential arm
   interleaves `(p0L1, p0L2, p1L1, …)`; the batched arm runs `(L1×K, then L2×K)`. A shared
   order-dependent state diverges by construction. Every real DeltaNet layer carries its own
   state; the harness now does too, and both recurrences went bit-identical.
2. **Raw relative error is meaningless at a near-zero reference.** Layer 2's worst raw relative
   error is 2.169e-3 — at a reference value of −8.2e-5. The same disagreement is 4.768e-7 on a
   2.276 range. The criterion now measures against the tensor's own magnitude and the receipt
   carries both figures plus the reference value at the worst point.
3. **The `valid` gate was left reading the raw figure** after the tolerance moved to the scaled
   one, so a correct run was marked INVALID and **the entire timing sweep it gates was silently
   skipped** — the receipt recorded `seq_gpu_ns_reps []` with no complaint. It surfaced only
   because analysing an empty list raised instead of returning a number. A gate that skips work
   on a false negative is worse than one that fails loudly.

A serial group was added around both arms while chasing (2). It changed nothing — the numbers
were identical to the digit across runs, so the discrepancy was never a race — but chained
dispatches inside one command buffer are now ordered explicitly rather than by assumption.

CP5a and CP5b carry the same raw-relative gate. Both measured `max_rel 0.0`, so it never
mis-fired there and their recorded results stand.

## The harness-level story is now complete

| question | answer | receipt |
|---|---|---|
| does batching win on one organ? | 2.474x q4, 2.49x affine gate_up, 2.27x affine down | CP3, CP4, CP4b |
| where is the knee? | (R=4,K=4); the governing variable is grid width vs core count, not registers | CP4b |
| do the organs compose in one command buffer? | yes, 2.458x, slightly above the perfect-composition expectation | CP5a |
| does the serial recurrence break it? | no. 2.205x, and it is cheaper inside the shared CB | CP5b |
| does a chunk cross a layer? | yes, ratio holds to −0.2%, no transpose needed | CP5c |

## What is still not done, stated plainly

**None of this is integrated.** All three kernel families have **zero call sites** in
`qwen38_hybrid_decode.rs`. Not measured: real artifact weight bytes at these shapes, residency
when every activation and temporary is multiplied by K across 64 layers, attention (6.5%,
needs chunked KV), and the complete prompt wall — which is the only number that settles whether
any of this is worth having.

What remains is not a measurement but an integration: teaching the runtime to step a CHUNK
instead of a token, at the per-token loop on `qwen38_hybrid_decode.rs:7566` and every organ
encoder it calls. That is runtime mutation, and under the standing single-writer rule it belongs
to HCLI.
