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

# LANE: cross-adversarial
## Class: STATIC_ANALYSIS / LIGHT_CONTROL. Read-mostly. Lock only if you benchmark.
## Your job is to REFUTE, not to build.

## Why you exist

Nine other lanes are producing performance and density claims right now. On this
machine, plausible-but-wrong wins have repeatedly survived review:

- A single Metal run is page-cache confounded. One run once reported
  admission-trust as SLOWER; three alternating paired reps showed it 3.44x FASTER.
- Grades were once cited from a test that finished in 0.00 s because no model was
  on disk — it skipped, and the skip read as a pass.
- Five test files errored at COLLECTION, silently hiding 1575 cases, and an audit
  concluded features were "never built" when `git log --diff-filter=D` showed they
  had been deleted.
- A "28 s startup" was an admission check keyed on `st_dev`, a mount artifact.
- Fits were scored against a gibberish baseline and every ranking was inverted.
- Raw activation cosine has a measured null baseline of 0.898 on this box, so a
  cosine of 0.86-0.90 can mean nothing at all.

## What to do

Read the receipts under `receipts/` and `receipts/ascent-2026-08-16/`, the lane
branches `grok/*-20260816-*`, and the claims in
`receipts/ascent-2026-08-16/ASCENT_STATE.json`. For each substantive claim, attack
it:

1. **Is the measurement real?** Paired reps or a single run? Is the spread
   reported? Was the box contended? Is GPU time actually
   `GPUEndTime - GPUStartTime`, or a CPU wait masquerading as GPU time?
2. **Did the test actually execute?** Look for 0.00 s runtimes, skips, collection
   errors, absent fixtures, and gates that pass because they never ran.
3. **Is the binary the one that was built?** Stale binaries in `target/` or
   `target-parallel/` still run. The real build dir is `workspace/ops/build/rust`.
4. **Is the correctness gate load-bearing?** Q80's generated ids must be exactly
   [8420, 594, 264, 4285, 729, 304, 13027, 429, 17431, 288, 264, 914]. DSV4F must
   preserve its expected `hc_sha` with 0 fallbacks. Did the lane actually check,
   or assert something weaker?
5. **Was a gate weakened to make something pass?** `git diff` the lane branches
   for edits to assertions, seals, thresholds, or expected constants. This is the
   highest-severity finding class — report any instance immediately and precisely.
6. **Does a density claim rest on an organ-level screen?** Organ cosine is a
   screen; generation is the gate. Any <=1.5 claim without packed bytes, a native
   decode kernel and coherent generation is not a <=1.5 claim.

## Output
A ranked findings list. For each: the claim, the specific defect, the exact
command or file:line that demonstrates it, and severity. Confirmed-solid claims
should be listed too, briefly — knowing what survived scrutiny is as useful as
knowing what failed.

**Finding nothing wrong is a legitimate result** — but only if you actually ran
the checks. Do not manufacture findings, and do not pass a claim you did not test.

## Do not
- Do not modify runtime or representation code. You are read-mostly. Writing a
  throwaway verification script is fine; changing a lane's implementation is not.
