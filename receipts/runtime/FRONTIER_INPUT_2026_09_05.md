# Measured frontier input — 2026-09-05

Everything here was measured this run, in this worktree. Ranked by share of the thing it sits in.

## Complete WorkUnit wall (one real WorkUnit, Mission.run path)

    total            549.959 s
    resident         547.362 s   99.5%   ONE model call
    persistence        0.0016 s    0.0%
    context_compile    0.0002 s    0.0%
    UNEXPLAINED        2.595 s     0.5%

Nothing outside the resident is worth optimising until the resident moves.

## Inside the GPU token (decode, all 64 layers; noop control 499 ns; 97% accounted)

    deltanet         7,929,749 ns   33.8%
    mlp_gate_up      7,269,166 ns   31.0%
    mlp_down         4,602,625 ns   19.6%
    gqa_attention    2,753,374 ns   11.7%
    lm_head            893,041 ns    3.8%
    embedding            5,791 ns    0.0%

## Standing measured facts

    decode            25.128 ms/token = 39.796 complete tok/s
    dispatches        916 / token
    KV                130,879 B/token, allocated f32 where the config says bfloat16
    recurrent+conv    156,893,184 B FIXED
    checkpoint        create 8.6 ms, restore 2.4 ms, break-even 0 positions
    non-GPU work      4.04% of the token wall
    thermal drift     +0.35% over hours -- anything smaller is not distinguishable
    batched prefill   2.116x at 1032 tokens, 35.3 -> 74.8 fresh tok/s, parity sha f36da4b00db5c5ed

## Open, unexplained

    One resident call took 547 s. Measured prefill at 74.8 tok/s puts a ~7K packet at ~90 s, and
    2048 output tokens at ~40 tok/s at ~51 s. Roughly 400 s is unaccounted INSIDE the call.
