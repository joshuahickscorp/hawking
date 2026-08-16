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
# LANE: dsv-integrate-occupancy
## Class: GPU_EXCLUSIVE for verification, COMPILE otherwise.
## INTEGRATION lane, plus ONE judgement call that is genuinely yours to make.

## Situation
`main` now carries the 3-way DSV4F integration (merge 80594ce4): cb-collapse
submission overhead + dsv-expert no-copy binds + dsv-mla simdgroup KV QAT, all
composed and bit-identical. Measured there: body 996 -> 830 ms, expert_slab_io
450 -> 239 ms, attention GPU 127.24 ms median, 43/43 no-copy binds.

NOT merged: `grok/dsv-occupancy-geometry-20260816-010044` (ahead=2), which attacks
the thread geometry dsv-mla identified (RMSNorm on 1 thread, realized 24-36 GB/s of
a ~750 GB/s ceiling):

    attention GPU  A 127.04 / 128.29 / 127.10 ms
                   B  52.57 /  58.02 /  53.62 ms
    cleanest pair  127.04 -> 52.57 ms; layer p50 2.90 -> 1.19 ms
    whole-token Metal GPU 330 -> 273 ms
    realized bandwidth now 92 GB/s of ~750

It touches matmul.metal, gravity_deepseek_v4_native_token_graph.rs,
gravity_deepseek_v4_token_ns_ledger.rs, metal/mod.rs - all of which main moved.

## THE JUDGEMENT CALL — do not paper over this
This lane is **NOT bit-identical**. The report states: *"SHA not bit-identical
(reduction order)"*, with token 5 logit 16.76857566833496 against the baseline
16.7818546295166, |delta| = 0.013279, and 0.001139 against the CPU oracle.

Every other DSV4F win so far preserved `hc_sha c94da765...` exactly. This one
changes it, because a parallel reduction reorders floating-point summation.

Your job, in order:
1. **Try to make it bit-identical.** A fixed reduction order or a deterministic
   binary-tree reduction usually recovers bit-identity at modest cost. Measure what
   that costs; if bit-identity is affordable, take it. This is strongly preferred.
2. If it is not affordable, present the numeric-equivalence case properly with
   evidence, not assertion:
       - per-layer hidden drift across all 43 layers, not just the final logit
       - logit drift and top-1 / top-5 agreement
       - drift against the CPU ORACLE (0.001139 is the meaningful number - it is
         SMALLER than against the native baseline, which suggests the new
         reduction may be CLOSER to true than the old one. Establish whether that
         is real; if so, say so plainly, because it reframes the change as an
         accuracy improvement rather than a regression)
       - a statement of what breaks downstream if hc_sha is no longer a stable
         identity for this route
3. Report a recommendation: BIT_IDENTICAL_ACHIEVED, NUMERIC_EQUIVALENT_ACCEPT, or
   HOLD. **Do not flip the default to a non-bit-identical path on your own
   authority.** If it lands, it lands behind an env flag, default off, until the
   numeric case is reviewed.

## Requirements
- Rebase/compose onto current main. NO WHOLESALE FILE COPY - main carries the
  3-way integration a blind overwrite would revert.
- The existing wins must survive: expert_slab_io ~239 ms, 43/43 no-copy binds,
  and attention GPU at least as good as the 127 ms baseline on the default path.
- 0 fallbacks.

## Negative science - do not re-pay
Collapsing the 17 attention dispatches into one serial encoder was bit-identical,
took encoders 731 -> 43, and did NOT move attention GPU. Encoder gaps are not the
attention wall. Occupancy is - which is why this lane exists.
