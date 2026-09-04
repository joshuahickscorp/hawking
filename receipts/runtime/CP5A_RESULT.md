# CP5a — the per-organ wins COMPOSE, measured in one command buffer

Raw: `receipts/runtime/CP5A_COMPOSITION.json`
Harness: `crates/hawking-core/examples/ascension_qwen38_cp5a_layer_composition.rs`

`CP5_DECOMPOSITION.md` projected 1.79-1.89x from batching the position-independent 78.7% of
the step, and listed what that projection did not prove. First on that list:

> that per-organ speedups compose. Organs measured alone do not share a command buffer, and
> production runs one CB per token.

This measures it. Both arms use a `TokenCommandBuffer`, which is the production shape.

* **SEQUENTIAL** — K command buffers, each holding the three organ dispatches for ONE
  position. That is exactly what the per-token prefill loop at
  `qwen38_hybrid_decode.rs:7566` does.
* **BATCHED** — ONE command buffer holding three multi-position dispatches covering all K
  positions.

Three weight-heavy organs at their real shapes, one layer's worth of each, at the r4k4 cell
that peaked independently in CP3, CP4 and CP4b:

| organ | codec | shape | kernel |
|---|---|---|---|
| in_proj | uniform q4 | 16384 x 5120 | `qwen_uniform_q4_group64_matmul_r4k4_geo_tpr64_tg128` |
| gate_up | affine q2 | 17408 x 5120 | `qwen_affine_q2_group64_matmul_gate_up_r4k4_geo_tpr64_tg128` |
| down | affine q2 | 5120 x 17408 | `qwen_affine_q2_matmul_r4k4_geo_tpr64_tg128` |

## Result

    command buffers   4  ->  1
    dispatches       12  ->  3
    GPU ns    1,264,957  ->  514,666

    SPEEDUP  2.458x  (ratio of medians)
             2.442x  (median of per-rep paired ratios)

    correctness  max_rel_err 0.0 across in_proj, gate, up and down -- BIT-IDENTICAL

9 reps, first discarded (it is an outlier in both arms: 2,138,831 and 886,083 against steady
~1,265,000 and ~515,000), zero loaded 27B residents by RSS, under `gpu_lane_lock.sh`.

**They compose, and slightly over-deliver.** Weighting the three organs by their step shares
and their individually measured speedups gives a perfect-composition expectation of **2.418x**;
the measured 2.44-2.46x is above it. The margin is the three command buffers the batched arm
does not pay for — organ measurements taken alone charge each dispatch its own CB, and the
composed form does not.

## What is still not proven

This is the composition claim and only that. Specifically outstanding:

* **Not a layer.** Norms, causal conv, recurrent state and attention are absent. The recurrence
  is the 9.8% that cannot be batched at all, and whether it forces a per-position boundary that
  re-serialises the projections around it is the next question, not this one.
* **Not the real weight bytes.** Both arms read the same deterministic buffers at real shapes;
  the q4/affine unpack is branchless so timing is content-independent, but no artifact segment
  was loaded here.
* **Not a prompt wall.** Still an isolated harness with zero call sites in
  `qwen38_hybrid_decode.rs`.
* **Residency untested.** Batching K positions multiplies every activation and temporary by K,
  and this harness allocates far less than a real 64-layer forward.

## Next

The one remaining structural question before integration: does a chunk of K positions survive
crossing a layer boundary, with the recurrence still stepping one position at a time, or does
the recurrence re-serialise the projections around it? That is the measurement CP5 proper owes.
