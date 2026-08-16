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

# LANE: q80-resident-table
## Class: GPU_EXCLUSIVE for benchmarks, COMPILE otherwise

## Measured baseline you must beat (taken 2026-08-16, DIRTY, one unrelated lane running)

Command:
    ./tools/gpu_lane_lock.sh q80-baseline \
      ./workspace/ops/build/rust/release-fast/examples/ascension_qwen80_uniform_q4_hybrid_greedy \
      --prompt "Write a function that reverses a string." --max-new-tokens 12 \
      --out receipts/ascent-2026-08-16/Q80_BASELINE_2026_08_16.json

Result:
    steady_state_tok_s = 2.479023   (403 ms/token)
    prefill_secs       = 11.158550
    decode_secs        = 4.437232 over 11 steady tokens
    stage_secs: deltanet=3.3269 gqa=1.1335 moe_table_build=9.0777
                moe_combine=1.6958  (others ~0)
    table_builds=1344  table_waves=1344  table_dispatches=6720
    fallback_count=1637 (vec=265 act=1344 sample=28)

## THE TARGET

`moe_table_build_secs` = 9.0777 s = **58% of the entire run**, across 1344 layer
builds (48 layers x 28 tokens) = **6.75 ms per layer build**.

Look at what one build actually does — `ensure_selected_expert_table` in
`crates/hawking-core/src/model/qwen80_uniform_q4_hybrid_decode.rs:743`:

- it fills a stack array `[Qwen80DeviceExpertTriplet; 512]` (512 * 128 B = 64 KiB),
- writes those 64 KiB into a Metal buffer via `write_buffer_bytes`,
- clones 6 `PinnedBuffer` handles for each of 10 experts into a `resources` Vec.

**6.75 ms to move 64 KiB is roughly three orders of magnitude off.** 64 KiB at even
10 GB/s is ~6 microseconds. So the time is NOT the bytes. Find out what it actually
is before you change anything. Prime suspects, in order:

1. `write_buffer_bytes` on a buffer the GPU still references -> implicit stall /
   wait-for-idle. Check the storage mode and whether anything synchronizes.
2. Buffer reallocation per call (`new_buffer_with_bytes_checked`) when the lease
   is not actually being reused.
3. `upload_qwen80_expert_triplet` cache misses still firing late in the run —
   instrument the hit/miss ratio, do not assume the cache is warm.
4. The 60 Arc clones + Vec growth per layer.

**Step 1 is measurement, not a rewrite.** Put a sub-stage breakdown inside
`ensure_selected_expert_table` (upload_miss_secs, entries_fill_secs,
buffer_write_secs, resource_clone_secs, lease_secs) and report the split. Do this
first and report the numbers even if you go no further — the split alone is worth
the lane.

## The structural fix (do this once you know the split)

The catalog is **40 GiB on a 96 GB machine**. It can be fully resident. There are
48 layers x 512 experts = 24,576 experts; the top-10 address table is rebuilt every
layer every token, which is artifact-static work being paid in the token loop
(this violates the standing rule: artifact-static work belongs in admission).

Target shape:

    ADMISSION (once):
        all expert triplets resident in device-addressable memory
        ONE address table covering all 48*512 entries, built once
    TOKEN LOOP:
        write 10 route ids (40 bytes)
        kernel indirects: addr_table[layer * 512 + route_id[k]]

That removes the per-layer 64 KiB rewrite AND the upload path AND the resource
clone entirely. `moe_table_build_secs` should approach zero.

Constraints to respect, and report on:
- Metal has a max buffer size (`device.maxBufferLength`) — check it on this M3
  Ultra. 40 GiB will almost certainly not fit one buffer. Use an `MTLHeap`, or a
  small set of large buffers plus an offset in the address entry. Report what you
  chose and why.
- `recommendedMaxWorkingSetSize` matters — report it and how close 40 GiB comes.
- If full residency does not fit, the fallback is a **persistent all-layer address
  table over a resident subset with an LRU for the tail**. Say so explicitly and
  measure the hit rate rather than quietly shipping a partial win.
- The existing `qwen80_compact_expert_slabs_enabled()` / `write_compact_selected_table`
  path and `write_top10_address_table` are two existing strategies. Read both, and
  say which one your change replaces or subsumes. Do not leave three live paths
  with no owner.

## Correctness gate
Generated token ids for the baseline prompt MUST stay exactly:
    [8420, 594, 264, 4285, 729, 304, 13027, 429, 17431, 288, 264, 914]
Any deviation is a failure, not a "numeric difference". Report the ids you got.

## Do not
- Do not touch the activation fallback path (act=1344). Another lane owns it.
- Do not touch DeltaNet or GQA. Another lane owns those.
- Do not change `QWEN80_UNIFORM_Q4_COMPLETE_PHYSICAL_BPW` or any seal constant.
