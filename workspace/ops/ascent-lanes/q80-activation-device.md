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

# LANE: q80-activation-device
## Class: GPU_EXCLUSIVE for benchmarks, COMPILE otherwise

## The target

The current Q80 baseline reports:

    fallback_count=1637 (matvec=0 embed=0 vec=265 act=1344 expert_bind=0 sample=28)

`act=1344` is exactly 48 layers x 28 tokens: **every layer of every token falls
back to a host activation**. The tournament constitution requires FALLBACK = 0, so
this is both a latency cost and a hard validity blocker.

Host activation classes are already broken out in `Qwen80ActivationClassTimes`
(`crates/hawking-core/src/model/qwen80_uniform_q4_hybrid_decode.rs:752`):
shared_swiglu, shared_mlp_sandwich, deltanet_conv, deltanet_recurrent,
gqa_input_layernorm, gqa_norm_rope, other_host_activation, metal_matvec_sync.

In the baseline run those all reported ~0 in the top-level `stage_secs` line, which
either means they are genuinely small or they are not being populated. **Establish
which, first.** Run with the ledger enabled:

    HAWKING_QWEN80_TOKEN_NS_LEDGER=1

(see `crates/hawking-core/src/model/qwen80_token_ns_ledger.rs:19`) and report the
per-class ns/token. A stage that reports zero because nothing writes to it is a
measurement bug worth fixing on its own.

## Prior work you MUST evaluate before writing anything new

Branch `grok/q80-activation-device-20260815-165122` contains a preserved,
NEVER-INTEGRATED lane: commit `71eeee25`, +1257 lines in
`crates/hawking-core/src/model/qwen80_uniform_q4_hybrid_decode.rs` plus
`receipts/QWEN80_UNIFORM_Q4_ACTIVATION_DEVICE.json`.

    git log --oneline main..grok/q80-activation-device-20260815-165122
    git diff main...grok/q80-activation-device-20260815-165122

Your first job is to judge it: does it actually move activations device-side, does
it build against current main, and does it hold the correctness gate? It was based
on an older main — **branch-skew rules apply: NO wholesale file copy.** Rebase,
graft minimally, or re-derive. Report which you did and why.

If it is good, land it. If it is broken or superseded, say exactly how and write
the correct version. Either outcome is a successful lane; silently ignoring it and
starting fresh is not.

## Correctness gate
Generated token ids for the baseline prompt MUST stay exactly:
    [8420, 594, 264, 4285, 729, 304, 13027, 429, 17431, 288, 264, 914]
and `fallback_count` act component must strictly decrease. Report both.

## Do not
- Do not touch `ensure_selected_expert_table` / the MoE address table. Another
  lane owns it. If your change needs something from there, say so in your report
  and work around it.
