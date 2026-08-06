# Single-device native execution track

This track is additive and default-off. It never writes the repository's live
`target/release/quantize-model`; every candidate build must use a directory below
`build/native-execution/`. Runtime defaults, Doctor queues, completed evidence, and
scientific results are unchanged.

Binary identity is intentionally explicit. At the 2026-07-14 audit checkpoint,
`target/release/quantize-model` is the root serial comparison binary with SHA-256
`79759a183604c826c83f25053a6be428b99ad5445ee43fd7919c485ec22ccf73`. The active
campaign quantizer is the separate
`build/strand-block-parallel/release/quantize-model-block-parallel` with SHA-256
`69ce7e09741e84a785604863f0fff369355c94185544646059baeeb08cabf4a9`. This native
scaffold reads or writes neither binary; it creates only fresh candidates below
`build/native-execution/`.

## Native CPU and PGO builds

`tools/native_build.py` confines target directories, raw/merged profiles, and every
receipt below `build/native-execution/`. It binds the host CPU, Rust toolchain,
source-file manifest, Cargo invocation, exact `RUSTFLAGS`, allocator identity,
target directory, and binary hash. The admitted native build is:

```sh
python3 tools/native_build.py build \
  --mode native \
  --target-dir ../../build/native-execution/cpu-native \
  --receipt ../../build/native-execution/cpu-native/build-receipt.json
```

PGO is a deliberately separated four-step workflow:

1. Build `--mode pgo-generate` with new raw-profile and target directories.
2. Only at an owner-free checkpoint, train that instrumented binary on the sealed,
   representative corpus. The run must emit a self-hashed
   `hawking.strand.native-pgo-execution.v1` receipt that binds the build receipt,
   exact program and argv/cwd/environment, sealed input and output bundles,
   owner-free lease-held admission, successful unskipped interval, physical raw
   profiles created during that interval, resource envelope, and exact non-skipping
   parity. Component fixtures and caller-declared hashes are not sufficient.
3. Seal that execution receipt with `seal-training`, then merge its exact raw profiles
   with the
   same instrumented-build and training receipts.
4. Build a new target directory with `--mode pgo-use`. Profile use revalidates the
   merged bytes, instrumented binary, training receipt, host identity, and complete
   source manifest before Cargo starts; any drift fails closed.

The authority-bearing commands are:

```sh
python3 tools/native_build.py seal-training \
  --instrumented-build-receipt ../../build/native-execution/pgo-gen/build.json \
  --execution-receipt ../../build/native-execution/pgo-gen/execution.json \
  --receipt ../../build/native-execution/pgo-gen/training.json

python3 tools/native_build.py merge \
  --instrumented-build-receipt ../../build/native-execution/pgo-gen/build.json \
  --training-receipt ../../build/native-execution/pgo-gen/training.json \
  --profraw ../../build/native-execution/pgo-gen/raw/default.profraw \
  --output ../../build/native-execution/pgo-gen/merged.profdata \
  --receipt ../../build/native-execution/pgo-gen/merge.json
```

Build targets and build/training/merge receipts are fresh and published with
exclusive no-replace semantics; an earlier evidence object cannot be overwritten or
a prior Cargo target silently reused. No build or PGO receipt activates a runtime.
Promotion still requires the complete
quantize-model exact-output receipt and tier/rate thread matrix.

The system allocator remains explicit in the receipt. No unreviewed process-global
allocator is installed. Current allocation tuning consists of exact-capacity input and
output buffers plus the already bounded pipeline and block scratch pools.

## Exact native I/O

`quantize-model-native --native-io preallocated|mmap` enables the native I/O adapter.
Both modes require `--native-staging-root` to name an existing nonsymlink directory
inside the repository's `build/native-execution/` boundary. Every dense/archive
output, generated JSON/de-bias sidecar, skip manifest, and index-stream output is
resolved and proven beneath that root before any input/model work starts. Primary
model/archive outputs must also end in `.partial`; suffix-only confinement is not
accepted.

- Preallocated input reserves the exact metadata length before reading.
- macOS mmap and preallocated-owned inputs bind device, inode, size, and nanosecond
  mtime. The opened file descriptor identity must equal the path identity before and
  after acquisition/read, and the retained identity is checked before and after
  finalization. A change fails closed.
- Every native archive, dense output, JSON/de-bias/index sidecar, and skip manifest is
  first written to a same-directory worker name created exclusively. The worker file
  is flushed and fsynced, its directory is synced, and publication at the requested
  `.partial` name uses an atomic no-replace hard link. The directory is synced again
  before and after removing the worker name.
- STR2 v2 remains invisible at its requested name while SDSC, OUTL, SDSQ, and SPRV are
  appended. The complete appended file is reopened and fsynced before publication, so
  a successful native return is a durable, complete staging artifact.

After exact archive, sidecar, source-identity, and receipt verification, an external
supervisor may promote the durable `.partial` artifact to its production name. This
code performs atomic staging finalization but does not perform or claim production
promotion. Rollback is the existing serial/default binary and removal of the
unpromoted partial candidate.

STR2 offsets, side sections, and SPRV still require all completed tensor records.
Therefore mmap/preallocation reduces copies and file growth but does not remove the
all-tensor finalizer working set. Pipeline record caps, completed results, input mapping,
pass-through tensors, and Viterbi scratch are reported separately.

## Metal RHT probe

`probe-metal-rht` contains a real 256-wide Metal rows/columns RHT adapter. Its strict
gate compares every output f32 bit against the CPU implementation. The safe command
beside a heavy owner is contract-only and performs no device creation or dispatch:

```sh
mkdir -p ../../build/native-execution/metal-contract
probe-metal-rht \
  --staging-root ../../build/native-execution/metal-contract \
  --receipt ../../build/native-execution/metal-contract/contract.json
```

Direct physical dispatch is inadmissible. At an owner-free checkpoint the only
admitted launcher is `tools/native_probe.py`: it acquires the shared heavy lease,
rechecks process owners under that lease, requires green RAM/swap/power/thermal state,
writes a fresh hash-bound admission receipt atomically inside the requested staging
root, and passes the still-held lease descriptor to the probe. The probe independently
checks receipt freshness, path confinement, receipt hash, descriptor/path identity,
and exclusive lock ownership before creating a Metal device.

```sh
python3 tools/native_probe.py \
  --staging-root ../../build/native-execution/metal-physical-<run-id> \
  --probe ../../build/native-execution/metal-probe/release/probe-metal-rht \
  --admission-receipt ../../build/native-execution/metal-physical-<run-id>/admission.json \
  --receipt ../../build/native-execution/metal-physical-<run-id>/parity.json
```

Even an exact synthetic
dispatch receipt leaves `runtime_activation=false`; production needs sealed real inputs,
both axes, exact output bundles, device/program/shader hashes, physical counters, and an
end-to-end quantize A/B. The Metal component result alone must never drive an ETA.

## Shared-power and thermal admission

Native/PGO training, Metal dispatch, and end-to-end A/B are admitted only when heavy
owners are zero, swap and disk headroom pass the existing guard, and the machine has a
recorded stable starting thermal state. CPU and GPU candidates run separately first:
Apple silicon shares memory bandwidth, package power, and cooling, so simultaneous
"maximum" CPU and GPU work can be slower and less reproducible. A stacked run is
admitted only after isolated receipts establish headroom and exactness.

Candidate receipts reserve separate fields for read/decode, RHT/preprocess, encode,
finalization/write, end-to-end wall, CPU/GPU time, peak RSS, swap delta, scratch peak,
disk bytes, and start/end thermal state. Missing fields remain `null` and make the
receipt non-promotable. Full-stack speed is evaluated only by matched end-to-end A/B;
component speedups are explicitly not ETA evidence.
