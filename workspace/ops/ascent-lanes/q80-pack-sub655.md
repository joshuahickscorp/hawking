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
# LANE: q80-pack-sub655
## Class: CPU_HEAVY / MEMORY_HEAVY, GPU under the lock for generation.
## Converts the strongest open candidate in the campaign into a real artifact or a real negative.

## Where this comes from
`q80-recalibrate-capability-bar` (merged) overturned the earlier sub-bit NO_GO by
showing the bar it was judged against was wrong.

    historical bar   0.8604   (D23 residual-identity break-even)
    corrected bar    0.6019   (last point that still GENERATED coherently)
    why              the packed 1.44445 BPW artifact generates coherent text with
                     down_proj holdout cosine 0.7684 - far under 0.8604. The screen
                     was wrong in the PESSIMISTIC direction.

Re-thresholded, the best sub-0.6552 recipe is:

    complete_physical_bpw   0.643662
    min organ cosine        0.673069   (above the 0.6019 corrected bar)
    fs/weight floor         98.2 fs    <- SUB-100

Its verdict is **ANALOG_COHERENT_NOT_PACKED**: it clears the bar under
cosine-matched degradation of the existing artifact, not as a packed artifact that
generated. That is a candidate, not a result.

## Your job
**Pack it for real and generate.** Recipe per the receipt's next step:
hgravs gate/up + binary down + ne4. Then:

1. Compute complete physical BPW from **bytes on disk**, the same way
   `q80-pack` did for mixed-1p5-v1 (which came out 1.44445 against a 1.44313
   design identity - count catalog, manifest, terminal, format and fit tables, do
   not exclude them).
2. Generate with the standard prompt and report the text VERBATIM.
3. Report fs/weight = 152.6252 * BPW / efficiency with the mandatory label
   AMORTIZED THROUGHPUT METRIC - NOT PHYSICAL SINGLE-WEIGHT LATENCY.

## The specific risk you must test, not assume
At that operating point the **second prompt already COLLAPSES** - the two-prompt
bar is 0.7684 and this recipe sits at 0.6731. So single-prompt coherence is likely
and multi-prompt coherence is doubtful.

**Generate on at least four varied prompts**, not one. Report each verbatim. A
recipe that answers one prompt and produces gibberish on the rest is NOT coherent,
and saying so plainly is the valuable outcome. Do not report the best prompt.

## Infrastructure limit already hit - work around it, do not rediscover it
A streamed RSS cap of 16 GiB killed the previous long session at 17,229,037,568
bytes after expert-cache growth. Run points in fresh processes, or bound the expert
cache. Do not let the pack die at 90%.

## Reference
    packer:    lab/operators/q80_mixed_representation_pack.py
    format:    docs/QWEN80_MIXED_1P5_PACKED_FORMAT.md
    catalog:   crates/hawking-core/src/model/qwen80_mixed_catalog.rs
    runtime:   crates/hawking-core/src/model/qwen80_mixed_hybrid_decode.rs
    prior art: mixed-1p5-v1 at 1.44445 BPW, GENERATES COHERENT

## Honesty
If it packs and generates coherently on varied prompts, this is the first sub-100
fs/weight artifact in the campaign and it should be reported as such, with its BPW
from real bytes. If it collapses, that is a clean measured NO_GO against a bar that
generation validated - which is worth far more than the previous NO_GO against a
bar generation had already contradicted. Either result closes the question.
