# TQ compute-for-memory appendix — 2026-07-14

This is Appendix B under `docs/plans/APPENDIX.md`.

## Relationship to the active ladder

This is an additive serving appendage to the existing compression/Doctor ladder,
not a replacement phase or a new artifact hierarchy. The ladder continues to
study bits per weight and Doctor recovery and remains the source of candidate
`.tq` artifacts. This appendix asks a later question about those same bytes:
which representation should the CPU/GPU reconstruct, gather, or recompute while
serving them?

The active Hawking run is read-only for this work. No process, queue state,
candidate, or current plan document is stopped or rewritten. Device performance
tests are deferred until the heavy owner exits.

## Audit result

The FLOPS path existed in pieces but was not a production serving choice:

- TQ/STR2 already had an exact deterministic computed Gaussian codebook mode,
  but scalar CPU decode materialized a full LUT before its hot loop.
- Metal had a computed decode-only kernel, while production fused GEMV always
  staged and gathered the stored codebook.
- `gate-coopwindow.rs` explored a state hash plus monotone quantile table, and
  `gate-tablecompact.rs` explored 84-byte to 40-byte block records. They remain
  scratch gates; no durable benchmark receipt was found, so they are hypotheses,
  not past wins.
- The expanded GPU block record is a hidden 84 bytes per 256 weights. That is
  2.625 metadata bits/weight before codebook staging, activation reads, partial
  reduction, RHT, and outlier correction. For a nominal TQ3 payload, the logical
  payload plus expanded runtime table is therefore 5.625 bpw. The 40-byte table
  reduces this to 4.25 bpw, saving 1.375 bpw of streamed metadata.
- The old fused reducer could silently mis-handle ragged/non-row-major block
  geometry. GPU admission now requires scalar row-major 256-weight blocks and
  `cols % 256 == 0`; unsupported geometry fails to the CPU path.
- An absent sub-scale side stream means exact unity. Two host bakes treated it
  like zero; both now use the canonical unity expansion.

## Built runtime paths

The `.tq` wire format and encoded symbols are unchanged. Runtime interpretation
is selected explicitly with `HAWKING_TQ_RUNTIME_PATH`:

| Mode | Runtime block table | Codebook source | Intended question |
|---|---:|---|---|
| `stored` | 84 B/block | state-indexed i32 Q12 LUT | proven baseline |
| `compact` | 40 B/block | state-indexed i32 Q12 LUT | is scale/min recompute cheaper than 44 metadata bytes? |
| `hashed` | 84 B/block | state hash + i16 monotone quantile table | is modest integer ALU worth halving codebook staging? |
| `computed` | 84 B/block | integer Acklam central quantile + exact stored tail | can a lookup-free codebook win in a genuinely bandwidth-starved shape? |

The CPU oracle now executes stored, hashed-quantile, and computed-Acklam sources
inside the decode loop instead of disguising computed mode as a prebuilt lookup.
The production Metal fused GEMV has corresponding stored, compact, hashed, and
computed kernels. Load-time selection is stored on `TqGpuReady`, so an autotuner
can compare explicit policies without mutating process-global environment state.
All four modes also have decode-only Q12 oracles and batch-major B=1..8 fused
kernels. The batch kernels decode each 256-weight block once and reuse it across
up to eight activations, preserving the same RHT-cols, OUTL, and residual recipe
components used by single-token TQ serving.

`TqRuntimeRecipe` separates metadata layout from codebook sourcing for accounting.
The four executable paths map into that grid; compact+hashed and compact+computed
are explicit research recipes but are not selectable kernels yet.

`TqRuntimeTraffic` exposes the logical payload bytes, expanded/compact table
bytes, per-threadgroup codebook staging, and the block-partial write/read
roundtrip. These counters are accounting, not a cache-hit or physical-bandwidth
claim.

Post-run device evidence has a fail-closed payload validator in
`tools/condense/tq_receipt_contract.py`. It requires eligible geometry, successful
Metal compilation, exact host/device record sizes, Q12 and fused-GEMV parity,
warm/trial floors, monotone latency percentiles, byte-accounting identity,
occupancy, realized bandwidth, energy, and the outer pressure/swap/thermal
receipt. A device microbenchmark is forbidden from requesting a default change.

## Cheap static census data

`tq_runtime_probe.py` now evaluates 1,036 analytic cells: 37 projection families
covering Q/K/V/O, FFN, and LM-head shapes from the locally present Qwen
0.5B-72B configurations plus a GPT-OSS-120B projection proxy, four symbol widths,
and seven current/future runtime recipes. It reads no model weights and makes no
physical-speed claim.

At TQ3, multiplicity-weighted compressed-path accounting shows approximately:

- compact metadata: **22.4-23.6% fewer listed runtime bytes** than expanded/stored;
- expanded hashed quantiles: about **2.0-2.2% fewer**;
- expanded computed Acklam: about **4.0-4.2% fewer**;
- future compact+computed: about **26.4-27.8% fewer**.

Those percentages include payload, metadata, per-threadgroup codebook staging,
and the block-partial roundtrip, but exclude activation/output bytes and do not
model physical cache hits. They make compact metadata the first device benchmark,
with codebook variants tested as incremental levers rather than assumed winners.

The same census exposes 168 ineligible cells across current/future recipes. The
affected model families are Qwen-0.5B (`hidden=896` projections), Qwen-72B
FFN-down (`intermediate=29568`), and the 120B proxy (`width=2880`). The strict
fallback is correct; a row-ragged fused kernel is now a named format/runtime
opportunity rather than an invisible loss of TQ GPU coverage.

`tq_runtime_matrix.py` converts the census into a dependency-closed post-run
queue: **496 executable deferred cells**, **96 implemented cells explicitly
blocked by geometry**, and **444 future-recipe design cells**. Every implemented
candidate depends on the matching stored baseline; no ineligible or not-yet-built
path is silently omitted or assigned a device receipt.

`appendix_postrun.py` audits the executable boundary around that queue. Six
existing vendor/static gates are mapped: stored/computed identity, compact
metadata, hashed quantiles plus cooperative windows, whole-token command-buffer
batching, staged decode writes, and their Rust compile surface. These are useful
candidate selectors, but several embed their own MSL, use synthetic/padded
geometry, report ad-hoc text or best-of-N timing, and do not exercise the
separate Hawking-core shader. They therefore cannot satisfy
`hawking.tq_runtime_device.v1`. The artifact adapter is now implemented by the
compiled `hawking-tq-device-probe` plus `appendix_device_runner.py`: it enforces
the shared lease, real-artifact hashes, exact Q12 and stored-GPU parity, resource
tripwires, and a separately bound physical-counter bundle. It has not been run
because the corpus still owns the machine, so there is still no device receipt.

The audit also exposed and fixed a launch-geometry nook: the single-vector
RHT-cols dispatch rounded a block count by multiplying it by 256, scheduling up
to 256 times more inactive grid positions than necessary. It now uses the same
threadgroup ceiling calculation as the TQ partial kernels.

## Why this belongs to the condenser mindset

Compression is not finished when the archive is small. A useful condenser owns
four coupled representations:

1. the durable symbol stream;
2. the metadata needed to make those symbols useful;
3. the runtime reconstruction algorithm;
4. the hardware schedule that turns reconstructed values into accepted work.

TQ can keep one durable representation while choosing a different point on the
bytes-versus-ALU frontier for each device and tensor shape. That makes the codec
an execution format rather than a file that must be fully dequantized before use.

Encoding is a different balance. The Viterbi/search path already spends much
more compute than decode and revisits codebook values, so materializing its small
exact LUT remains sensible unless an encode-specific profile disproves that.
The FLOPS experiment is primarily a serving/decode lever.

## Post-ladder gate matrix

Run only under the same exclusive heavy/GPU lease as Doctor and quantization.
For every stored/compact/hashed/computed cell:

1. compile the Metal source and verify host/device record sizes;
2. assert exact Q12 decode equality across tail-biting, affine/non-affine,
   sub-scale-present/absent, and supported L/K shapes;
3. assert fused GEMV parity against the stored path and the CPU Q12 matvec;
4. measure warm p50/p95/p99 wall time, energy, payload/table/codebook/partial
   bytes, memory pressure, swap, and thermal state;
5. sweep projection geometry, L, K, number of blocks, and threadgroup residency;
6. reject any cell with an output mismatch, swap growth, abnormal pressure, or a
   confidence interval that includes no improvement.

No runtime mode becomes default from a microbenchmark alone. It must win accepted
tokens per second or joules per accepted token in native `.tq` serving, including
RHT, outlier correction, activation traffic, partial reduction, and command
overhead.

## Research synthesis

The closest prior art supports the direction but also sharpens the gate. QTIP
explicitly describes a spectrum from lookup-only to computed lookup-free trellis
codes rather than one universally optimal decoder
([QTIP](https://arxiv.org/abs/2406.11235)). That is the architectural basis for
keeping runtime policy off-wire and per-device.

The strongest counterpoint to “compute everything” is FLUTE: its reported gains
come from offline weight restructuring that reduces awkward bit manipulation,
plus LUT vectorization/duplication that avoids shared-memory bottlenecks
([FLUTE](https://arxiv.org/abs/2407.10960)). Therefore the post-ladder comparison
must include a fifth family beyond the four current modes: **layout-repacked
stored lookup**. It may beat full recomputation by making the lookup path
hardware-native. QuIP# likewise couples randomized Hadamard incoherence with
hardware-efficient lattice codebooks, reinforcing that transform, codebook, and
kernel are one design problem rather than independent knobs
([QuIP#](https://arxiv.org/abs/2402.04396)).

Apple's Metal guidance makes occupancy a first-class acceptance metric: register
and threadgroup-memory pressure can reduce the number of concurrent threads, and
memory-bandwidth counters must be interpreted together with occupancy and limiter
counters
([Metal compute optimization](https://developer.apple.com/videos/play/tech-talks/10580/),
[GPU occupancy](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy),
[memory bandwidth](https://developer.apple.com/documentation/xcode/measuring-the-gpus-use-of-memory-bandwidth)).
The computed Acklam kernel therefore needs instruction/register/occupancy evidence,
not just fewer source bytes. Within the codebook axis, the hashed i16 route is the
lower-compute candidate; across the complete runtime path, the static census says
compact metadata has the much larger byte opportunity. Both remain hypotheses
until device receipts exist.

## Next compute-for-memory nooks

- Combine compact metadata with hashed/computed codebooks after isolating each
  lever; the first implementation keeps them orthogonal for attribution.
- Build a row-ragged fused path for the measured 896/29568/2880 column families,
  while retaining strict admission as the correctness oracle.
- Remove or hierarchically reduce the f32 partial write/read roundtrip. It is
  eight logical bytes per block and can erase codebook savings.
- Test cooperative state windows and multiple outputs per thread only after
  state-chain dependency, occupancy, and register pressure are visible.
- Specialize affine-free tables so unused min metadata disappears rather than
  streaming zeros.
- Search load-time table formats per tensor shape, including a smaller seek table
  plus separately packed scale/min streams.
- Add FLUTE-style offline runtime repacking and LUT duplication as a control; do
  not assume recomputation dominates an efficiently laid-out gather.
- Fuse activation RHT or reuse transformed activations when one input feeds
  multiple projections; count any extra activation rereads.
- Treat outliers as a joint format/schedule decision. Sparse corrections help
  distortion but can introduce random reads, atomics, and another dispatch.
- Build a byte-roofline receipt for every projection: useful MACs, integer ops,
  compressed bytes, metadata bytes, activation bytes, scratch bytes, occupancy,
  and realized bandwidth. “More FLOPS” is useful only when it removes the
  bottleneck byte stream.

## Files in this slice

- `crates/hawking-core/src/tq_gpu.rs`: runtime policy, compact bake, strict GPU
  admission with machine-readable rejection reasons, orthogonal recipe/traffic
  accounting, upload selection.
- `crates/hawking-core/shaders/strand_bitslice.metal`: compact, hashed-quantile,
  and computed-Acklam fused GEMV kernels.
- `crates/hawking-core/src/kernels/mod.rs`: runtime kernel dispatch and parity
  gate scaffold.
- `vendor/strand-quant/src/{codebook,decode,trellis}.rs`: real CPU source modes.
- `vendor/strand-decode-kernel/src/block_walk.rs`: canonical absent-sub-scale
  handling.
- `tools/condense/tq_runtime_probe.py`: artifact-free projection and runtime
  recipe census; its generated report/receipt live under `reports/appendix/`.
- `tools/condense/tq_runtime_matrix.py`: stored-baseline-bound post-run device
  queue with deferred, geometry-blocked, and design-deferred states.
- `tools/condense/appendix_postrun.py`: fail-closed gate inventory and dependency
  bridge from corpus freeze through TQ device evidence into speculative decode.

This slice establishes correctness and experimental control. It deliberately
makes no speed claim before device receipts exist.
