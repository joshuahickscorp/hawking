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

# WHY THIS LANE SURVIVES THE ARTIFACT CHANGE — read this first

Q80's uniform-Q4 vehicle (4.259241 BPW) has been **abandoned as a target**. Q80 is
being re-gravitied to a <=1.5 complete-physical-BPW artifact in parallel lanes.

**Your lane is artifact-INDEPENDENT.** The cost you are attacking is a property of
the token graph, the dispatch topology, the state handling or the host/device
split — not of how expert weights are encoded. It will still be there, essentially
unchanged, on the <=1.5 artifact.

So: use the Q4 catalog purely as a **test harness and correctness reference**. It
is a convenient runnable vehicle, nothing more.

**The design rule that follows from this:** do not hard-code anything to the
uniform-Q4 representation. Where you touch a path that knows about weight
encoding, keep the mechanism generic so the <=1.5 artifact inherits it for free.
If you cannot avoid a Q4-specific assumption, isolate it behind a seam and say so
in your report. A win that has to be rebuilt for the real artifact is a half win.

Q80 measured baseline on the Q4 harness (2026-08-16, DIRTY, 12 new tokens):
    steady_state 2.479023 tok/s = 403 ms/token; prefill 11.16 s
    stage_secs over the 15.6 s run:
        moe_table_build 9.0777 (58%)   deltanet 3.3269 (21%)
        moe_combine 1.6958 (11%)       gqa 1.1335 (7%)
    fallback_count=1637 (vec=265 act=1344 sample=28)
    table_builds=1344  table_dispatches=6720
    generated ids: [8420, 594, 264, 4285, 729, 304, 13027, 429, 17431, 288, 264, 914]

Target is a 20 ms complete token (50 TPS). Everything above is ~20x too slow.

---

# LANE: q80-runtime-residency
## Class: GPU_EXCLUSIVE for benchmarks. THIS LANE IS "STRUCTURE FOR THE FUTURE".

## The finding you are generalizing

`moe_table_build_secs` is 58% of the Q80 run: 1344 layer-builds (48 layers x 28
tokens) at **6.75 ms each, to write a 64 KiB address table**. 64 KiB is
microseconds of bytes at any plausible bandwidth. **The cost is not the bytes.**

Entry point: `ensure_selected_expert_table`,
`crates/hawking-core/src/model/qwen80_uniform_q4_hybrid_decode.rs:743`.
It fills `[Qwen80DeviceExpertTriplet; 512]` (512 * 128 B), writes it via
`write_buffer_bytes`, and clones 6 `PinnedBuffer` handles per expert into a
`resources` Vec. Also read `write_top10_address_table` and
`write_compact_selected_table` in `qwen80_device_expert_table.rs` — two existing
strategies whose relationship you should clarify.

**Step 1 is measurement.** Add a sub-stage split (upload_miss / entries_fill /
buffer_write / resource_clone / lease) and report it. Prime suspects: an implicit
stall because the GPU still references the buffer being written; per-call
reallocation rather than true lease reuse; late `upload_qwen80_expert_triplet`
cache misses (instrument hit/miss, do not assume the cache is warm); Arc-clone
churn. Report the split even if you go no further — it is worth the lane alone.

## The general law you are implementing

    artifact-static work  -> admission (paid once)
    session-static work   -> setup
    token-dynamic work    -> the token loop, and nothing else

Rebuilding an address table every layer every token is artifact-static work being
paid per token. The target shape:

    ADMISSION: expert payloads resident and device-addressable;
               ONE address table covering all 48*512 entries, built once
    TOKEN:     write the selected route ids (tens of bytes)
               kernel indirects: addr_table[layer * n_experts + route_id[k]]

**Build this as a general residency + indirection mechanism, not a Q4 special
case.** The <=1.5 artifact will have different payload formats (binary_group,
binary+rice residual, low-rank factors) but the SAME need: resident payloads, one
static address table, per-token indirection. Parameterize over payload layout so
the new artifact drops in. This generality is the actual deliverable.

## Physical constraints to measure and report
- `device.maxBufferLength` on this M3 Ultra, and `recommendedMaxWorkingSetSize`.
- The Q4 catalog is 40 GiB; the <=1.5 artifact will be far smaller, so residency
  gets *easier* later. Design for the harder case, report both footprints.
- If full residency does not fit, use a persistent all-layer address table over a
  resident subset with an LRU tail, and **measure the hit rate** rather than
  quietly shipping a partial win.

## Correctness gate
Generated ids must stay exactly the 12 listed above. Report what you got.

## Do not
- Do not touch the host-activation fallback path; another lane owns it.
- Do not touch DeltaNet/GQA; another lane owns those.
