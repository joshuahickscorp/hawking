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

# LANE: dsv-resident-gravity
## Class: CPU_HEAVY / MEMORY_HEAVY. Use the GPU lock for any GPU or >8 GiB step.

## The thesis: for DSV4F, density IS velocity

The source artifact is ~148 GiB. The machine is an M3 Ultra with 96 GB unified
memory. The model therefore **cannot be resident**, and the entire streamed-source
runtime regime — reader, mmap, page faults, per-token payload staging, overlapped
prefetch threads — exists only because of that.

A <=1.5 BPW representation would put DSV4F in a resident-class footprint and
**delete that whole regime** rather than optimizing it. This is architectural
replacement, not storage polish. That is why this lane exists.

## Your job this lane is the honest feasibility answer, with numbers

1. **Geometry.** Get the real parameter census: total params, per-organ breakdown
   (experts / shared expert / MLA / attention / embeddings / norms / lm_head).
   Compute what 1.5 BPW, 1.4, and 1.3 complete-physical each imply in GiB.
   Complete physical means everything: codes, scales, codebooks, rank factors,
   indices, padding, and any side table — not just the code payload.
2. **What has to be non-expert.** Q80's answer was a *mixed per-component* policy:
   1.43051 complete BPW with experts compressed hard and non-experts left at full
   8-bit. Establish the analogous split for DSV4F: what fraction of DSV4F's mass
   is routed-expert, and does the same structure hold?
3. **Residency arithmetic.** A resident model still needs headroom for KV / MLA
   latent state, activations, scratch, and the OS. State the real budget, not just
   model bytes. Say plainly whether <=1.5 BPW actually lands under 96 GB with
   working room, and at what margin.

## Transferable science from Q80 — use it, do not re-derive it
- Mixed per-component representation is REQUIRED; a single codec family across all
  components is insufficient. gate_proj / up_proj / down_proj prefer different
  families and down_proj *inverts* the ranking (low-rank beats binary there).
- down_proj must be fit against **post-SwiGLU X**, not the layer hidden. Fitting it
  against the hidden is a known wrong-target error.
- Experts are mutually orthogonal (cos 0.004) — a shared cross-expert basis is
  REFUTED. Do not propose one.
- Rice+q1 residual coding cost 8.24 bits/outlier versus 48 for the naive form.
- Fits must be calibrated on activations captured from the **real BF16 model**, not
  from a degraded baseline and not from synthetic/Gaussian activations. Both of
  those have produced total false negatives in this codebase.
- A fit with fewer captured rows than the tensor's dimension is UNDERDETERMINED and
  its score is meaningless. Report rows-per-fit against dimension for anything you
  fit, and treat `rank = min(budget, n_fit_rows)` style caps as a red flag.

## Deliverable
A written feasibility verdict with the arithmetic behind it, plus whichever of
these the evidence supports: a capture plan, a per-organ codec proposal, or an
explicit statement that <=1.5 BPW resident DSV4F is not reachable and why.

**A well-evidenced negative is a successful lane.** Do not manufacture optimism.
Do not declare success from organ-level estimates — at model level the gate is
coherent generation, and nothing short of that counts.

## Do not
- Do not modify the runtime. Other lanes own it. This lane produces evidence,
  a plan, and at most offline packing/analysis tooling.
- Do not create a giant JSON index. A 1.38 GB capture-result.json was a real
  iteration wall here; use a compact/CSR/mmap form.
