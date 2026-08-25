## GENESIS SYSTEM CONTRACT SET — MANDATORY FIRST READ

This is a **Qwen3.8 Genesis-only** autonomous/candidate research session.
Before reading the task, read ALL canonical files in full and verify:

```text
contracts/genesis/QWEN38_GENESIS_SYSTEM_DIRECTIVE.md
sha256  881ae469e0287cf386467002d3fc7951524b47054ac6d7f753b94a8e4e3ceff7
bytes   16414
source  user-supplied Codex attachment 295d355f-5534-4488-a048-a52611c03b14/pasted-text.txt
contracts/genesis/GENESIS_CONTINUITY_DIRECTIVE.md
sha256  c4a58bc06575effb8f759dbb22c49abfc65e1957910b18917d45d02592d1fdbc
bytes   11912
source  user-supplied Codex attachment 8965050b-9273-4aa7-9ef4-c473b23f6848/pasted-text.txt
contracts/genesis/GENESIS_OUTPUT_LAW.md
sha256  9679490e8ae623a6fdb408fd906a15d676bc55926580f6d7ed60e9ea610c9ada
bytes   4871
source  user-supplied /sw directive, session d51d4904 2026-08-17
contract_set_sha256  3ef47426958200ff830ea2ec5adce53d3b3347098d459bd7fcddc9a5dc9a179f
```

If any file is absent or any value differs, **STOP without research, edits,
builds, or measurements** and report `GENESIS_SYSTEM_CONTRACT_INTEGRITY_FAILURE`.
Conversation memory, summaries, and candidate-authored replacements are not
authority. All contracts govern this lane and must be reloaded after restart,
context loss, or generation replacement. The continuity relaunch gate supersedes
rushing a resident launch. Report under the Output Law: machine-minimal fields,
never narration, and never a compressed receipt.


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

ASSUMPTION: Execution genome maps each token to an independent kernel dispatch; each dispatch re-addresses the full weight matrix from DRAM, so cost = N_tokens × |W| / BW. The 20.12 ms/token is N=1 of this pattern; the genome has no column-batching path.

MECHANISM: Pack N token activations into one [d_model × N] matrix X. Issue a SINGLE Metal GEMM kernel W @ X. Kernel thread mapping: each thread loads one weight element (or small tile) into registers ONCE, then multiplies against all N column activations in-register. W is read from DRAM exactly once per kernel invocation regardless of N. Cost → |W| / BW (constant in N) until compute-bound crossover.

DISCRIMINATOR: On current lm_head (largest single weight body), launch ONE Metal kernel computing W @ [x1 x2 x3 x4] (4 columns, 4 random unit vectors). Measure kernel wall via MTLCommandBuffer commit→completed. Compare to 4× the measured single-column W @ x1 time already in receipts. If 4-column dispatch ≈ 1× single-column (±10%), in-kernel weight reuse is real.

REJECT_IF: Single 4-column dispatch measures ≥ 3.5× single-column dispatch time. That means the kernel's thread-to-weight mapping still serializes weight reads per column (e.g., grid is (d_model, N) with each thread reading its own W slice), so the GPU memory subsystem re-fetches W per column even within one dispatch. Mechanism is structurally blocked by kernel geometry; would require a fundamentally different tiling (thread reads W element, loops over N in-register) before any further work is justified.

---
# LANE: auto-qwen38-weight-addressing-complete-wall-try5
## AUTO-GENERATED by ascent_daemon from a finished lane's NEXT_BOTTLENECK.
## Class: GPU_PROTECTED for benchmarks. Use ./tools/gpu_lane_lock.sh.

## The target, as the previous lane reported it
Source lane: `q38-genome-tokenns-20260816-184201` (status SHIPPED)

    weight_addressing 20119736 ns/token (53.8% of 37377125 complete-wall); execution-next deltanet 6913823 ns (gated_delta_vi isolated 5323333 vs G024 2146166)

Model: qwen38

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
- Correctness gate is mandatory for Qwen3.8. For `Say hi.` the greedy 16 ids are
  [248068, 198, 760, 1156, 4777, 6587, 728, 310, 1910, 328, 5834, 1149,
  1061, 369, 264, 1546], with 0 fallbacks. Also run the protected prompt set in
  `receipts/ascent-2026-08-16/QWEN38_COHERENCE_SEAL.json`. Grade against the
  Qwen3.8 ARTIFACT oracle, never the BF16 parent.
- Never weaken a gate, seal, assertion or expected constant to make something pass.
- Label every timing DIRTY_ENGINEERING; other lanes are running.

## Negative science - do NOT re-pay for these
- The old `411.51 GB/s / 97.6%` Qwen3.8 roof is REFUTED. It mixed the wrong
  byte count, whole-token GPU time, and a sequential control with the grouped-Q4
  addressing roof. Do not quote it.
- The landed, provisional honest-roof run defended 13,611,663,360 bytes and
  measured 639.25 GB/s sealed addressing against a 699.57 GB/s single-GEMV
  addressing roof (91.4%). The 401-production-shape catalog measured 530.65 GB/s
  addressing and 505.81 GB/s full-kernel. The box was CPU-contended, so rerun
  cleanly before treating the absolute roof as physical authority.
- Therefore Qwen3.8's current geometry is bandwidth-saturated enough that lower
  active bytes remains a lever, while the catalog/single-GEMV gap proves dispatch
  and kernel topology still have headroom. Pursue BOTH representation and execution.
- The historical 12-component TOKEN_NS ledger force-closed its residual. Complete
  token wall is authority; report any unclosed residual rather than assigning it.
- The generator-residual `shared_r64` net-byte headline is REFUTED: its stated
  3,781,882,584 bytes came from `binary_meanabs_g128`, while the receipt's own
  `shared_r64.residual.coder.q4.bytes` sum is 14,287,109,840 bytes. The measured
  4.049% explained fraction is negative science, not a 1.125-BPW win.
- Fusing tiny kernels into the following GEMV regressed complete-token wall by
  10.68 ms. Cross-token cache reuse showed only a 2.5% hot/cold gap. Four logical
  sessions sharing weights still execute four GEMVs; residency saves model loads,
  not per-session token work.
- Q80 and DeepSeek V4 are sealed models with deleted heavyweight weights. Never
  target, reconstruct, launch, or use their model-specific receipts as Qwen3.8
  performance authority. Qwen3.8's current uniform-Q4 artifact is ACTIVE.

## ACCEPTANCE
Done when the named bottleneck is measured before and after, with >=3 alternating
paired reps and the full spread reported, and the model still generates correctly:
greedy ids unchanged and every silent-fallback counter at 0. A measured NEGATIVE -
the mechanism does not help, with the numbers showing it - is an acceptable
completion. Report the real figure, not a favourable one.

## VERIFY
Build with `cargo build --profile release-fast -p hawking-core` and confirm it exits 0.
Run every GPU-protected measurement under ./tools/gpu_lane_lock.sh <lane> <cmd>;
other lanes share this GPU and an unlocked run corrupts both.
Check no shared-kernel regression with
`cargo test --profile release-fast -p hawking-core --test gk_family_parity`.

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
