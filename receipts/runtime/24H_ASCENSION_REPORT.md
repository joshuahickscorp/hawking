# 24H ASCENSION REPORT

2026-09-05. Isolated worktree `.worktrees/ascension`, branch `ascension-isolated`.
Every number below traces to a receipt produced this run, in this worktree, on this machine.
Nothing projected is reported as achieved; projections are labelled and separated.

## Commits

    starting commit   403b13bf1   (worktree pin; main checkout released to a concurrent refactor)
    ending commit     see `git log --oneline 403b13bf1..HEAD`

## Benchmark table — all MEASURED

### Decode (supersedes both standing receipts)

    complete wall        25.128042 ms/token  =  39.7962 complete tok/s
    GPU                  24.113999 ms/token
    wall - GPU            1.014043 ms         =  4.04% of the token
    six alternating rep medians 25.082-25.150 ms, 0.27% spread
    uninstrumented control 25.168 ms/token, agreeing to 0.16%
    FALLBACKS 0
    prior standing: 26.17 complete tps, and a newer 25.671. Both superseded.

### Dispatches

    916 per token, measured, on both cold-or-first and steady decode.
    Resolves a three-way disagreement in which NONE of the standing numbers was right:
    964 (derived: 1 embed + 64x15 + 3 terminal, never measured), 628 (live HCLI), 708 (CP6d).

### Per-organ decode share (noop control 499 ns; organs sum to 97% of the measured GPU token)

    deltanet         7,929,749 ns   33.8%
    mlp_gate_up      7,269,166 ns   31.0%
    mlp_down         4,602,625 ns   19.6%
    gqa_attention    2,753,374 ns   11.7%     (carried at 6.5% before this run)
    lm_head            893,041 ns    3.8%
    embedding            5,791 ns    0.0%

### Prefill — batched GEMM vs sequential, capability-equivalent

    tokens   sequential    batched     ratio    seq tok/s   bat tok/s
        11      342.3ms     742.1ms    0.461x       32.1        14.8
        73     1903.4ms    1504.4ms    1.265x       38.4        48.5
       265     6885.8ms    3840.1ms    1.793x       38.5        69.0
      1032    29206.3ms   13802.2ms    2.116x       35.3        74.8

    Parity at 1032: both arms byte-identical, sha f36da4b00db5c5ed, 146 non-empty bytes.
    The comparison carries a negative control proving it detects a one-word mutation.
    P50 CLEARED. P60 CLEARED. P75 is 74.8 against a 75.0 bar and is NOT claimed.

### Context / memory

    hw.memsize                96.0 GiB (probed)
    resident weights          10,554,259,456 B (disk payload reconciles to the byte, difference 0)
    attention KV              130,879 B/token measured by RSS slope; 131,072 read from code (99.85%)
                              the long-carried 65,536 B/token is off by 200%; cause is dtype (f32
                              allocated where the config says bfloat16)
    recurrent + conv          156,893,184 B FIXED, does not scale
    prefix checkpoint         that state + 8 B, in-process only, no serializer
    peak RSS                  10.890 GiB @512 .. 12.825 GiB @16384 max_seq_len
    swap                      Swapouts DELTA 0 across a full benchmark
                              (the raw counter reads 1,227.8 GiB and is a boot high-water mark)
    262,144 projection        31.95 GiB KV + 10.83 base = 42.78 GiB on a 96 GiB machine -- PROJECTED
                              from the measured slope, NOT run.

### Checkpoint economics

    recompute 256 positions   6,133,635,000 ns  (23,959,511 ns/position)
    CREATE                        8,585,958 ns
    RESTORE                       2,427,208 ns
    NET SAVED, one hit        6,122,621,834 ns
    BREAK-EVEN                            0 positions

### Thermal control

    same workload hours apart: 39.796 -> 39.934 complete tok/s, drift +0.35%.
    DECISION RULE: any promotion under ~0.35% is not distinguishable from the machine.
    Everything promoted this run clears that by orders.

## Accepted mutations

    - typed-tool WorkUnits survive persistence (tool/tool_arguments were dropped by every reload)
    - context budget consults the native artifact ceiling (claimed 32768 against a real 8192)
    - generation reserve clamped to what the engine can spend (2048 stranded tokens returned)
    - WorkUnit.author recorded, defaulting to unrecorded rather than to "hcli"
    - complete-WorkUnit and mission wall decomposition with an honest UNEXPLAINED bucket
    - batched GEMM prefill landed and its shader made to compile (14/14 kernels resolve)
    - batched prefill wired into generate_greedy_complete_wall
    - flash attention wired into the Qwen3.8 path
    - mha_decode_f32 over-budget dispatches routed to flash instead of returning garbage
    - prefix checkpoint refuses restore across reset()

## Rejected / declined on measurement

    - KV compression (G008). KV traffic sits in an 11.7% organ; a perfect 2x is worth <=5.9% of
      the decode wall, while DeltaNet is 33.8%. Declined, with a reopen condition.
    - CPU/GPU overlap bounties (G011). Non-GPU work is 4.04% of the token; the SMALLEST listed
      bounty is 5%. All five exceed the entire pool. Arithmetic, not difficulty.

## Two silent-wrong-answer defects found and fixed

Both completed without error. Neither had a guard. This is the failure class that costs most,
because nothing looks broken.

    mha_decode_f32   indexes scores[seq_len] in threadgroup memory. Past the 32,768 B device limit
                     Metal returns NO error and the numbers are wrong:
                        seq  8192   34,816 B  over budget, still correct (4.8e-7)
                        seq 16384   67,584 B  DIVERGE  rel 1.65e2
                        seq 32768  133,120 B  DIVERGE  rel 3.08e2
                     A hard ceiling would have been safe. This returned confident nonsense.

    checkpoint       carries rec_state and conv_state but NOT the KV cache, and reset() zeroes it.
                     Restoring across a reset gave next token 4242 against 358 for the honest walk.

## Laws

    L1  A dispatch that exceeds its threadgroup budget does not fail. It returns a wrong answer.
    L2  complete_tps amortises prefill, so it falls with prompt length BY CONSTRUCTION. A thermal
        control must be the SAME workload, not any workload.
    L3  A bare prompt is NOT a WorkUnit. controller.execute calls engine.execute directly; Mission
        is only built by run_mission.
    L4  Field names are part of the instrument. A census read off guessed keys reports a
        plausible zero.
    L5  Measured thermal drift on this machine is +0.35% over hours of sustained load.

## Scars

    S1  Standing "attention is 6.5% of the step" -- measured 11.7%.
    S2  Standing "KV is 65,536 B/token" -- measured 130,879. Off by 200%.
    S3  Standing "964 dispatches/token" -- derived, never measured. Actual 916.
    S4  Census claim "the threadgroup limit caps context at 7,680 tokens" -- REFUTED. There is no
        cap; the kernel never refuses.
    S5  Standing "44 of 55 specimens lack a manifest" -- now 6.
    S6  dense_w_materialized is a structural constant: account_dense_w has zero call sites, so
        every receipt reporting 0 reports nothing.

## Autonomy accounting -- the honest number

    HCLI_GENERATED_WORKUNITS / ALL = 0 / 0 for optimization work this run.
    CLAUDE_CAUSAL_WORK = all of the accepted mutations above.

    C1 (>=50% HCLI-generated) is NOT met and is not close. S001 established a supervisor-only law;
    S004 then directed that limits be engineered away rather than catalogued, and the engineering
    was done by Claude. That is a real regression against the autonomy objective and it is
    reported, not folded into the runtime results.

    The instrument for measuring it now exists (WorkUnit.author, authorship_report, defaulting to
    unrecorded so legacy units cannot inflate the ratio). The measurement does not.

## Remaining bottleneck

    DeltaNet decode, 33.8% of the GPU token -- larger than either MLP half. Every prefill result
    this run targeted the MLP. CP6 deliberately left the recurrent mixer per-position because it is
    sequential; that is right for prefill, where positions are known ahead, and irrelevant to
    decode, which has one position and pays the 33.8% every token.

## Diminishing returns

    Not reached. §7 requires >=3 distinct experiments AND >=2 hours in a family with neither >1%
    complete-wall improvement nor a new discriminator. The prefill family produced 2.116x; the
    attention family produced a correctness fix; the measurement family produced five superseded
    standing figures. No family is exhausted.

## Next highest-leverage frontier

    1. DeltaNet decode (33.8%, measured, untouched)
    2. G001's wall decomposition through a real Mission -- the instrument is wired and the path is
       now known (/mission, not a bare prompt)
    3. C1 autonomy: HCLI generating its own optimization WorkUnits, which no runtime result
       substitutes for

## Not done, stated plainly

    - 262K context NOT run. The 42.78 GiB figure is a projection from a measured slope.
    - ANE-1 (proving actual ANE execution) open. MLComputePlan reports the PLAN; corroboration
      needs powermetrics ane_power (root) or xctrace (absent).
    - HCLI Anywhere (G013) and the governed web tool class (G014) untouched this run.
    - The batched prefill 2.116x is a single pair per length, not CP6i's order-alternated protocol.
