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

# LANE: dsv-admission-identity
## Class: IO_HEAVY / CPU_HEAVY. Lock for GPU or >8 GiB steps.

## The measured target

From `receipts/ascent-2026-08-16/DSV4F_HOST_WALL_BASELINE.json` (baseline
AUTHORITY, 6 reps):

    host_exclusive                        564.9 ms/token median
    validation_identity.sha_ns            0            <- SHA already eliminated
    validation_identity.identity_ns             (parallel sum)   71.9 ms
    validation_identity.path_resolve_ns         (parallel sum) 1318.0 ms
    validation_identity.verify_ns               (parallel sum) 2505.5 ms
    address_lookup.tensor_lookup_calls    140,261 BTreeMap probes (~12 ms)

The lane's own note: *"path_resolve (checked_regular_path + file_identity stat) is
the dominant identity tax and runs BEFORE the digest cache. 3314 calls, ~400 us
each in parallel-sum."*

These are parallel sums across threads, so they do not add directly to the
critical path — **do not report a parallel sum as if it were token latency.** Your
first job is to determine how much of it is actually ON the critical path. But
3314 path resolutions per token, each doing a `checked_regular_path` plus a
`file_identity` stat, is immutable-artifact work being repaid every token.

## The standing law you are enforcing

    cold artifact proof  ->  sealed session proof  ->  token path trusts the seal

Integrity stays strong; only the repetition disappears. Never repay per token:
SHA, geometry parsing, manifest scan, directory walk, identity derivation.

SHA was already eliminated on this path (sha_ns = 0), which is proof the pattern
works here. `path_resolve` + `stat` is the same class of tax, and it runs BEFORE
the digest cache, so the cache never gets a chance to help.

## Strong prior from this machine
Immutable-identity recomputation has been the top latency finding **three separate
times**: a per-token SHA, a clone-tree-on-open, and an admission check keyed on
`st_dev` — a mount artifact — that alone cost 28 s of startup and, once dropped,
took startup 13.5 s -> 2.3 s and a warm repack 4606 s -> 95 s (48x). Look for the
same shape: identity derived from something that cannot change within a session,
recomputed anyway.

## Do this
1. Attribute path_resolve/verify to the critical path versus overlapped work.
   Report both; do not conflate them.
2. Move what is genuinely artifact-static into admission behind a session seal.
3. Also assess the 140,261 per-token `tensor_lookup` BTreeMap probes (~12 ms).
   A precomputed dense index resolved once at admission likely replaces the whole
   structure. 140k probes per token for a fixed tensor set is artifact-static work.

## Correctness gate
The seal must remain load-bearing. Do not delete verification — move it. Expected
`hc_sha` preserved (baseline `c94da765`), 0 fallbacks. State exactly what is now
verified once instead of per token, and what still guards it.

## Do not
- Do not touch expert streaming (`dsv-expert-cache`), CB topology
  (`dsv-cb-collapse`), or MLA (`dsv-mla`).
