# STRAND execution and promotion guide

STRAND supplies deterministic archive/quantization paths used by Hawking. The
production rule is simple: acceleration may change scheduling and I/O, never
arithmetic, search order inside a block, tensor order, packed bytes, scientific
receipts, or completed evidence.

## Execution paths

| Path | Purpose | Default |
|---|---|---|
| serial | canonical comparison and fallback | on |
| block-parallel | independent trellis blocks across bounded CPU workers | off |
| ordered pipeline | overlap read/preprocess, encode, and ordered sink | off |
| native I/O | preallocated or mmap input plus durable partial output | off |
| native/PGO | host-specific CPU candidate | off |
| Metal RHT | bit-exact preprocessing probe | off |

Every candidate builds below an isolated directory such as
`build/strand-block-parallel/` or `build/native-execution/`. Never replace a
binary used by an active cell.

## Block-parallel encoder

The `block-parallel` Cargo feature preserves each block's arithmetic and search
order, then gathers packed payloads in source order. It forces one outer tensor
worker so tensor fanout cannot multiply block workers. Aggregate Viterbi scratch
is capped; the canonical single scratch buffer remains the irreducible floor.

```sh
CARGO_TARGET_DIR=build/strand-block-parallel \
  cargo build --release --features block-parallel \
  --bin quantize-model-block-parallel \
  --bin gate-block-parallel \
  --bin gate-quantize-model-block-parallel

STRAND_NO_GPU=1 \
  build/strand-block-parallel/release/quantize-model-block-parallel ... \
  --threads 1 --block-threads 20 \
  --block-scratch-budget-bytes 268435456
```

Promotion requires exact parity across two-pass PSI, tail biting, affine minima,
adaptive scales, scalar/vector trellises, LUT and codebook modes, f32 metric and
search modes, dense tensors, sidecars, and complete packed-v2 archives.

## Thread profiles

Production selection is exact for `(tier, rate)`. Each key requires authenticated
8, 12, 16, and 20-thread receipts from the same source, binary, mode, scratch
budget, and canonical output. The fastest candidate inside the RSS ceiling wins;
ties choose fewer threads. There is no nearest-tier, nearest-rate, or nominal
fallback.

```sh
python3 vendor/strand-quant/tools/thread_profile_contract.py build \
  --receipt 72b-q3-8.json --receipt 72b-q3-12.json \
  --receipt 72b-q3-16.json --receipt 72b-q3-20.json \
  --expected-binary-sha256 "$BINARY_SHA" \
  --rss-limit-bytes "$RSS_LIMIT" \
  --output thread-profile.json
```

Synthetic fixtures prove mechanics only. A production profile needs real,
source-bound artifacts, exact output, wall time, and peak RSS.

## Ordered bounded pipeline

The ordered path overlaps:

1. source read and deterministic preprocessing;
2. caller-owned encoding, optionally block-parallel;
3. an ordered output sink.

Both boundaries are bounded synchronous channels. Records carry ordinals and
conservative resident-byte accounting. Order mismatch, budget violation, stage
error, panic, or early closure fails the call. The final STR2 container still
requires all tensor results because offsets, side sections, and the outer seal
are global; the pipeline bounds in-flight records rather than claiming a
streaming finalizer.

A sink writes a fresh temporary artifact. Publication occurs only after
pipeline completion, exact-output verification, and receipt finalization.

## Native CPU, PGO, and I/O

`vendor/strand-quant/tools/native_build.py` confines target directories,
profiles, and receipts beneath `build/native-execution/`. Receipts bind the host
CPU, toolchain, source manifest, Cargo invocation, flags, allocator, target, and
binary.

PGO is four separate authorities:

1. build an instrumented candidate;
2. run it on a sealed representative corpus at an owner-free checkpoint;
3. seal the execution and merge the exact raw profiles;
4. build a new profile-use target and repeat output/receipt parity.

No build receipt activates a runtime.

Native I/O requires an existing nonsymlink staging root inside
`build/native-execution/`. Primary outputs use `.partial`, are written through
exclusive same-directory worker files, flushed and fsynced, and published with
atomic no-replace semantics. Input identity is checked through the open file
descriptor before and after use. Production promotion remains external.

## Metal RHT

The Metal RHT adapter must compare every output f32 bit with the CPU
implementation. Contract-only checks may run beside a heavy owner, but physical
dispatch requires the shared heavy lease, a fresh direct owner census, green
memory/swap/power/thermal probes, and an admission receipt that remains valid
while the device is created.

A component parity receipt does not activate the runtime or change ETA.
Production needs sealed real inputs, both axes, exact output bundles,
device/program/shader hashes, physical counters, and matched end-to-end A/B.

## Resource admission

Apple silicon shares RAM, bandwidth, package power, and cooling. Qualify CPU and
GPU candidates separately before stacking them. Candidate receipts keep
read/decode, preprocessing, encode, finalization/write, total wall, CPU/GPU
time, peak RSS, swap delta, scratch peak, disk bytes, and thermal start/end as
separate fields. Missing physical fields make the receipt non-promotable.

## Archive integration

`.sa` is a deterministic solid archive. Extraction rejects absolute paths,
drive/root components, and `..` traversal. The CLI registration surface is:

```sh
strand register
```

- macOS uses a headless signed `STRAND.app`;
- Linux installs per-user MIME and desktop associations;
- Windows uses a windowless opener shim when available.

Release packaging and credentials remain deployment concerns, not repository
documentation. A distributable macOS bundle must be signed, notarized, stapled,
verified, installed, registered, and smoke-tested. Do not record private keys,
passwords, or machine-local secret paths in documentation.

## Operational pattern

Long STRAND runs use a stateless-resumable conductor:

- derive truth from checkpoints and filesystem state each tick;
- apply mechanical corrections before waking judgment;
- suppress repeated low-salience events without suppressing escalation;
- run cheap invariant replay only while idle;
- never change science flags, resource caps, or defaults from the watchdog;
- avoid `pgrep -f` self-match by bracketing one character;
- append machine-readable metrics and keep Markdown out of the hot loop.

## Promotion checklist

1. Exact serial/candidate output and receipt parity.
2. Complete real-artifact tier/rate thread matrix.
3. Owner-free process, lease, RAM, swap, disk, thermal, and power admission.
4. Matched randomized end-to-end A/B, not a component extrapolation.
5. Crash/resume and rollback proof.
6. Pending-only source-bound runtime generation.
7. Atomic promotion at a quiescent checkpoint.
8. Serial fallback retained until the new generation is independently verified.
