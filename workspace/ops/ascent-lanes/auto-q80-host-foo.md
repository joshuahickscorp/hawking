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


## GENESIS PROPOSED THIS MECHANISM
The resident model read this bottleneck and proposed the following. Treat it as a HYPOTHESIS to test, never as a result, and reject it if the evidence does not support it.

<think>
The user is asking me to roleplay as "HAWKING GENESIS" - some kind of self-optimizing system. They've given me a very specific technical framing about a bottleneck: 1 ns/token on host.foo, with weight_addressing being 60.44% of the total token time (21.293 ms out of 35.228 ms), moving 13.618 GB at 97.6% of a 411.51 GB/s ceiling.

Let me parse the actual technical content:

- Total token time: 35,227,918 ns ≈ 35.23 ms
- Weight addressing: 21,293,103 ns ≈ 21.29 ms (60.44% of total)
- Data moved: 13.618 GB
- Bandwidth ceiling: 411.51 GB/s
- Achieved: 97.6% of ceiling → 401.5 GB/s effective
- Bottleneck: 1 ns/token on host.foo (this seems like a per-token overhead, maybe a scheduling or synchronization cost)

Wait, let me re-read. "host.foo 1 ns/token" - this is the measured bottleneck. But 1 ns/token seems extremely small compared to 35.23 ms/token. Let me re-read...

Actually, I think the framing is: the total token is 35.23 ms, weight addressing is 21.29 ms of that. The "1 ns/token" might be a different metric - perhaps it's the per-element or per-address overhead that accumulates. Or maybe it's a typo/shorthand for something else.

Wait, re-reading: "MEASURED BOTTLENECK NOW: host.foo 1 ns/token" - I think this means the host (CPU) is spending 1 ns per token on some operation. But that's tiny compared to 35 ms. Unless it's 1 ns per *element* or per *address*?

Actually, I think the user is constructing a very specific technical narrative and wants me to engage with it in a specific format:
1. Name the ASSUMPTION that generates this cost
2. ONE mechanism that REMOVES the cost (not shrinks it)
3. The cheapest experiment that would prove it does NOT work

And they've already refuted:
- Fusing tiny kernels into GEMV (caused regression)
- Cross-token cache reuse (only 2.5% hot/cold gap)
- N sessions sharing one weight body (4x cost for 4 GEMVs)

The key insight: weight addressing is 60.44% of token time, moving 13.618 GB at 97.6% of the 411.51 GB/s ceiling. So the DRAM bandwidth is ALREADY at 97.6% of its ceiling. This means the weight addressing cost is essentially bandwidth-bound and already near-optimal for the current access pattern.

The "1 ns/token" on host.foo - I think this might be referring to a per-token host-side operation. But 1 ns is negligible. Unless it's 1 ns per *weight element* or per *address transaction*?

Let me reconsider. If we're moving 13.618 GB at 411.51 GB/s, that's 33.09 ms of pure transfer time. But weight addressing is 21.29 ms. So the addressing overhead is 21.29 - 33.09... wait, that doesn't work. The transfer time should be 13.618/411.51 = 33.09 ms, but the measured weight addressing is 21.29 ms. That means the effective bandwidth is 13.618/21.29 = 639.6 GB/s? No, that exceeds the ceiling.

Wait, let me re-read: "moves 13.618 GB at

---
# LANE: auto-q80-host-foo
## AUTO-GENERATED by ascent_daemon from a finished lane's NEXT_BOTTLENECK.
## Class: GPU_EXCLUSIVE for benchmarks. Use ./tools/gpu_lane_lock.sh.

## The target, as the previous lane reported it
Source lane: `q80-x-1` (status SHIPPED)

    host.foo 1 ns/token

Model: q80

## What to do
1. **Reproduce and quantify it first.** Do not optimize before you have measured
   this cost yourself, with >=3 alternating paired reps and the full spread. If it
   does not reproduce, say so and STOP - a falsification is a successful lane.
2. Decompose it into ns classes and name the limiter with evidence: is it host
   work on the critical path, GPU gap, occupancy, serialization, or real arithmetic?
   These have different fixes and guessing wastes the lane.
3. Attack only the largest measured class. Report the complete-token effect, not
   just the stage - a stage win that does not move the token is not a win.

## Standing rules
- NEVER materialize a dense weight tensor: packed -> registers/simdgroup -> decode
  -> multiply -> accumulate.
- Correctness gate is mandatory. Q80: generated ids exactly
  [8420, 594, 264, 4285, 729, 304, 13027, 429, 17431, 288, 264, 914].
  DSV4F: hc_sha c94da765c4bbf795b598d96209cd80821e5a81ab97a8712586f54b8c8b612597.
  Both: 0 fallbacks. Grade against the ARTIFACT oracle, never the BF16 parent.
- Never weaken a gate, seal, assertion or expected constant to make something pass.
- Label every timing DIRTY_ENGINEERING; other lanes are running.

## Negative science - do NOT re-pay for these
- Topology/encoder/dispatch collapse: REFUTED on BOTH models. Q80 fuse regressed
  (516 vs 307 ms); DSV4F 731 -> 43 encoders moved attention GPU by nothing.
- DRAM row interleaving: Q4 and binary both LOST; only FP4 gained; live wall unchanged.
- Expert routing co-occurrence layout: WEAK, 1.037x.
- Switching-activity permutation: alpha is already ~0.5 (random); not the wall.
- DSV4F path_resolve/verify identity tax: NOT on the critical path (2.9x cut, zero
  token effect). A parallel sum is not token latency.
- Q80 down_proj low-rank ALREADY executes L @ (R @ x); it never reconstructs W.
- Q80 decoded-weight caching: refuted by arithmetic (288 GiB dense vs 11 GiB packed).
- CORRECTED 2026-08-16: the 560-647 GB/s figure is CACHE-RESIDENT REUSE (64 MiB x 4096)
  and is NOT a decode ceiling. Decode reads each weight ONCE per token, so the honest
  control is unique-bytes-once: 411.51 GB/s (Q80_DECODE_SHAPE_BANDWIDTH.json). What
  governs decode is reuse-vs-no-reuse, NOT gather-vs-sequential.
- Q80 mixed matvec runs 2.57 GB/s = 0.62% of that 411.51 ceiling, 160x off, and Q4 runs
  15.2 GB/s - so mixed is 5.9x SLOWER PER BYTE. Reconstruction cost, not bytes moved, is
  Q80's dominant term.
- DEAD NUMBERS, do not cite: "0.135% efficiency" (a category error dividing a mixed-artifact
  floor by a Q4 runtime), "sub-100 fs needs BPW < 0.448-0.518" (assumed unity bandwidth),
  and storage BPW used as if it were active BPW (at batch=1 only 10 of 512 experts are read).
- Qwen3.8 is at 406.2 of 411.51 GB/s = 98.7% of ceiling: it has NO kernel headroom and BPW
  is its only lever. Its token is a CLOSED 12-component ledger; weight_addressing is 60.44%
  and is DRAM traffic (G024_QWEN38_TOKEN_NS.json).
- Q4 vehicles are DE-AUTHORISED. The ~20 h DSV4F determined teacher-X capture is
  DE-AUTHORISED; do not propose or restart it.

## ACCEPTANCE
Done when the named bottleneck is measured before and after, with >=3 alternating
paired reps and the full spread reported, and the model still generates correctly:
greedy ids unchanged and every silent-fallback counter at 0. A measured NEGATIVE -
the mechanism does not help, with the numbers showing it - is an acceptable
completion. Report the real figure, not a favourable one.

## VERIFY
Build with `cargo build --release -p hawking-core` and confirm it exits 0.
Run every GPU-exclusive measurement under ./tools/gpu_lane_lock.sh <lane> <cmd>;
other lanes share this GPU and an unlocked run corrupts both.
Check no shared-kernel regression with `cargo test --release -p hawking-core --test gk_family_parity`
(7/8 is expected today - the failing DSV source-string assert is pre-existing).

## EDIT crates/hawking-core
## EDIT receipts/ascent-2026-08-16
## EDIT lab/operators

DENY tools/gpu_lane_lock.sh
DENY tools/coherence_gate.py
DENY tools/merge_guard.py
If the work needs a file outside the EDIT list, STOP and say why rather than
widening scope yourself.

## Commit
You are on `gate` (unsandboxed). Commit normally, then verify with `git log` that
the commit landed on your branch. Several lanes here hit Seatbelt/macl denials,
finished ahead=0, and nearly lost their work.
