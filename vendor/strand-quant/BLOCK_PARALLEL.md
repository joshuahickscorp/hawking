# Feature-gated block-parallel encoder

The `block-parallel` feature adds an explicit CPU path that distributes independent
trellis blocks across scoped threads and gathers their packed payloads in original
block order. Arithmetic and search order inside each block are unchanged. Existing
`encode_tensor*` entry points and the default quantizer path remain serial.

## Isolated build

Build into a target directory that is not used by the live campaign:

```sh
CARGO_TARGET_DIR=/Users/scammermike/Downloads/hawking/build/strand-block-parallel \
  cargo build --release --features block-parallel \
  --bin quantize-model-block-parallel \
  --bin gate-block-parallel \
  --bin gate-quantize-model-block-parallel
```

The accelerated quantizer is then:

```text
/Users/scammermike/Downloads/hawking/build/strand-block-parallel/release/quantize-model-block-parallel
```

Its block path is opt-in and fail-closed:

```sh
STRAND_NO_GPU=1 quantize-model-block-parallel ... \
  --threads 1 \
  --block-threads 20 \
  --block-scratch-budget-bytes 268435456
```

`STRAND_NO_GPU=1` is mandatory because this is parity against the canonical CPU
explicit-LUT path. The binary forces one outer tensor worker in block mode so tensor
fanout cannot multiply the requested block workers. The scratch budget caps aggregate
worker Viterbi scratch; one canonical scratch buffer remains the irreducible floor.
Retained completed paths are immediately bit-packed rather than held as `u32` symbols.

One-pass rolling PSI is rejected because each block depends on earlier entropy.
Two-pass PSI, tail biting, affine minima, adaptive scales, scalar/vector trellises,
custom LUTs, stored/hashed/computed codebooks, and f32 metric/search modes are covered
by exact-parity tests.

## Promotion gates

1. Run `block_parallel_parity` with and without `STRAND_F32_METRIC=1` and
   `STRAND_F32_SEARCH=1`.
2. Run `gate-block-parallel` at the intended thread and scratch settings. Its
   `hawking.strand.block-parallel-parity.v1` receipt binds the gate binary, inputs,
   serial/parallel encoded hashes, timings, and canonical payload digest.
3. Run `gate-quantize-model-block-parallel` against a current-source serial build and
   the feature build. Its
   `hawking.strand.quantize-model-block-parallel-parity.v1` receipt binds both binary
   hashes and requires exact dense safetensors, JSON sidecar, and packed-v2 archive
   hashes on a generated multi-tensor canary.
4. Separately compare the packed-v2 and dense hashes with the source-bound live
   quantizer. Historical binaries can have an older JSON sidecar schema even when
   their scientific outputs are identical, so raw legacy-sidecar equality must not be
   synthesized.
5. Promote only into pending specs at a quiescent checkpoint; never replace or mutate
   the source-bound binary of an active cell.
