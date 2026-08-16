## HAWKING ASCENT — standing laws for every lane

Repo: /Users/scammermike/Downloads/hawking . Build dir is `workspace/ops/build/rust`
(set by .cargo/config.toml). NEVER use `target/` or `target-parallel/` — stale
binaries there still run and have produced false results before.

Build: `cargo build --profile release-fast -p hawking-core --example <name>`
Binaries: `workspace/ops/build/rust/release-fast/examples/<name>`

### Resource discipline (MANDATORY)
This machine runs several lanes at once. Any command that touches the GPU or
allocates more than ~8 GiB MUST be wrapped:

    ./tools/gpu_lane_lock.sh <your-lane-name> <your command...>

It is a mutex; it blocks until free (90 min cap). Compiling, reading, static
analysis and unit tests do NOT need it. Never bypass it — an unlocked benchmark
run silently corrupts another lane's timing.

### Measurement law
- A single Metal run is page-cache confounded. Any timing claim needs >= 3
  alternating paired reps (A,B,A,B,A,B) and you must report the full spread,
  not just the median.
- GPU time means `MTLCommandBuffer.GPUEndTime - GPUStartTime` after wait.
  A CPU wall-clock wait is NOT GPU time; never report it as such.
- Label every number DIRTY_ENGINEERING (other lanes running), CLEAN_CANDIDATE,
  or BASE_TRUE. Do not launder a dirty number into a clean claim.
- Report ns/token, not just tok/s.

### Correctness law
- Bit-identity or a stated numeric-equivalence gate is required for every
  optimization. "Looks close enough" is a rejected result.
- 0 fallbacks. If a fast path silently falls back, that run is invalid.
- Never weaken an existing gate, assertion, or seal to make something pass.
  If a gate blocks you, report it as a finding — do not edit it away.

### Negative science — do NOT re-pay for these
- Q80 cross-expert shared-basis: REFUTED (experts mutually orthogonal, cos 0.004).
- Q80 "simply bandwidth-bound": REFUTED. Measured 0.79% of the 700-800 GB/s
  ceiling with ~51% GPU idle. It is dispatch/host bound, not bandwidth bound.
- DSV4F route-ID readback serializer hypothesis: REFUTED.
- Shader compile as the primary current wall: REFUTED / deprioritized.
- Single-family Q80 representation: INSUFFICIENT. gate_proj/up_proj/down_proj
  each prefer a different codec family; down_proj inverts the ranking and needs
  post-SwiGLU X, not the layer hidden.
- Q30 static <=1.5 coherence: FAILED. Do not copy the Q30 approach.
- Immutable-identity recomputation (SHA, st_dev, geometry parse, manifest scan)
  per token has repeatedly been the real latency. Suspect it early.
- Giant JSON indexes are a real iteration wall (1.38 GB capture-result.json).
  Do not add one.

### Reporting
End your final message with:

    LANE: <name>
    STATUS: SHIPPED | PARTIAL | BLOCKED
    BASELINE_NS_PER_TOKEN: <n> (label)
    RESULT_NS_PER_TOKEN: <n> (label)
    REPS: <the actual paired numbers>
    CORRECTNESS: <bit-identical | numeric gate + measured drift | N/A>
    FILES: <paths touched>
    RECEIPT: <path to json receipt you wrote under receipts/ascent-2026-08-16/>
    NEXT_BOTTLENECK: <what is now the top cost, with its measured ns>

Commit your work on your branch before finishing. Uncommitted lanes have been
lost here before.

---

# LANE: dsv-integrate-graph
## Class: GPU_EXCLUSIVE for verification, COMPILE otherwise.
## This is an INTEGRATION lane. The science is done; do not redo it.

## The situation

Two DSV4F lanes independently restructured the SAME MoE dispatch path in
`crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs`. One is
already merged to `main`; the other is not. A plain merge produces 4 conflicts,
two of which are large (114 vs 48 lines, and a 262-line block), because both
rewrote the same `submit`/encode region for different reasons.

    ALREADY ON MAIN   grok/dsv-cb-collapse-20260816-001742  (merged)
        one ordered compute encoder per command buffer (encoders 1857 -> 134)
        six expert act_quants batched into one dispatch
        hash layers 0-2: upload experts, then route+moe in one CB
        adds counters: total_sync_points/readbacks/buffer_creations/rebinds,
                       scratch_buffer_creations, cb_collapse_enabled(),
                       GpuGapAccounting
        MEASURED: GPU 412 -> 377 ms, bit-identical

    NOT YET MERGED    grok/dsv-expert-20260816-003212  (commit 858894e3)
        replaces host slab PACKING (76 MiB/layer write_at) with NO-COPY binds:
        pin the six mmap'd expert-chunk windows as MTLBuffers, write a six-entry
        gpuAddress table, let dsv4f_worklist_fp4_matvec index them (useResources
        required). Hash layers 0-2 bind during attention GPU.
        adds counters: expert_nocopy_binds, expert_slab_packs, expert_payload_path
        env A/B: HAWKING_DSV4F_EXPERT_SLAB_PACK=1 restores the old pack
        MEASURED same-binary A/B: expert_slab_io 481 -> 262 ms, body 1278 -> 1016 ms
        43/43 no-copy binds, bit-identical


    ALSO NOT MERGED   grok/dsv-mla-20260816-003211  (commit 1983ec3b)
        (a) one serial encoder for the 17 attention dispatches: encoders 731 -> 43,
            bit-identical, but attention GPU DID NOT MOVE (195-220 ms both arms).
            NEGATIVE RESULT, kept for the encoder reduction only. Encoder gaps are
            not the attention wall - do not re-pay for that hypothesis.
        (b) simdgroup KV QAT on the same E4M3FN table encoder: isolated kv_qat
            2.9 ms -> 107 us per layer. THIS is the real win.
        MEASURED attention GPU, GPU-locked alternating pairs:
            198.32 -> 127.84 ms | 196.97 -> 161.74 ms | 269.40 -> 168.68 ms
        touches: shaders/matmul.metal, src/metal/mod.rs,
                 gravity_deepseek_v4_native_token_graph.rs, ..._token_ns_ledger.rs
        MLA is OCCUPANCY-limited, not bandwidth-limited: 24-36 GB/s of a ~750 GB/s
        ceiling. RMSNorm runs on 1 thread, WQ-A 1024, WKV 512, sparse attn 64.

**All three lanes touch the same graph and ledger files.** Land all three. They are
complementary: submission overhead (cb-collapse), host payload staging
(dsv-expert), and attention occupancy (dsv-mla) are three different costs.

**All changes are wanted. They are complementary, not competing:** one reduces
encoder/submission overhead, the other removes host payload staging. Your job is
to land the second on top of the first with both behaviours intact.

## Do this

1. Rebase or hand-compose `858894e3` onto current `main`. **NO WHOLESALE FILE
   COPY** — main has instrumentation from `dsv-host-wall` and `dsv-cb-collapse`
   that a blind file overwrite would silently revert. That is the specific failure
   this lane exists to avoid.
2. Keep BOTH counter sets and BOTH JSON receipt fields. Conflicts 1 and 2 are
   purely additive; take both sides.
3. Conflicts 3 and 4 are the real work: the no-copy address-table bind must be
   expressed inside the collapsed single-encoder-per-CB topology, not alongside a
   resurrected multi-encoder path. If the two are genuinely incompatible in some
   respect, say so explicitly and explain the tradeoff rather than quietly
   dropping one.
4. Keep BOTH env escapes working, since they are the A/B harness:
       HAWKING_DSV4F_CB_COLLAPSE=0        -> pre-collapse topology
       HAWKING_DSV4F_EXPERT_SLAB_PACK=1   -> old host pack
   Verify all four combinations at least build, and that the two defaults-on
   paths are correct.

## Proof required — this is the whole point of the lane

    hc_sha == c94da765c4bbf795b598d96209cd80821e5a81ab97a8712586f54b8c8b612597
    greedy token 5, logit 16.7818546295166 (delta 0.0144 <= 0.05)
    fallbacks == 0
    43/43 no-copy binds reported
    attention GPU at or below the ~128 ms the mla lane measured
    expert_slab_io at or below the ~262 ms the expert lane measured

on the integrated build, plus paired A/B reps showing the expert_slab_io win
SURVIVED integration. A merge that compiles but loses the win, or that regresses
`hc_sha`, is a failed lane. Report the measured numbers, not "should be
equivalent".

## Measurement honesty
The box is heavily loaded; everything is DIRTY_ENGINEERING. Use
`./tools/gpu_lane_lock.sh` and paired alternating reps, and report the full spread.
Note that the prior lane saw one pair flip (no-copy 1404 ms vs pack 1278 ms) — a
flipped pair in a 3-pair set is expected noise on this box, so report it rather
than hiding it, and do enough reps that the median is meaningful.

## Commit
You are on `gate` (unsandboxed). Commit normally and then verify with `git log`
that the commit actually landed on your branch. Earlier lanes hit Seatbelt denials
writing to `.git`, finished `ahead=0`, and nearly lost their work.
