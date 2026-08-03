# Receipt-gated thread profiles and ordered pipeline

This layer is additive and disabled by default. It does not change the live
`quantize-model` entry point, the existing block-parallel binary, completed evidence,
or any runtime specification.

## Per-tier/rate thread selection

`tools/thread_profile_contract.py` builds and verifies exact-match thread profiles.
For every exact `(tier, rate)` key, production calibration must provide all four
candidate receipts: 8, 12, 16, and 20 threads. Candidates must bind the same source,
binary, canonical output, scratch budget, and execution mode. Every output must equal
its canonical hash. The fastest candidate that also satisfies the profile RSS limit is
selected; ties prefer fewer threads.

There is deliberately no nearest-tier or nearest-rate fallback. A missing, synthetic,
non-exact, stale, tampered, mixed-source, mixed-mode, or incomplete receipt matrix is
ineligible.

A production candidate receipt uses this schema:

```json
{
  "schema": "hawking.strand.tier-rate-thread-canary.v1",
  "status": "pass",
  "scope": "production",
  "synthetic": false,
  "tier": "72B",
  "rate": "q3",
  "threads": 16,
  "binary_sha256": "<64 lowercase hex>",
  "source_sha256": "<64 lowercase hex>",
  "canonical_output_sha256": "<64 lowercase hex>",
  "output_sha256": "<same canonical hash>",
  "exact_output": true,
  "wall_seconds": 123.45,
  "peak_rss_bytes": 45678901234,
  "scratch_budget_bytes": 268435456,
  "mode": "block_parallel"
}
```

For `mode: "ordered_pipeline"`, the candidate must additionally bind
`pipeline_receipt_path` and `pipeline_receipt_sha256`. That pipeline receipt must itself
be a passing `hawking.strand.quantize-model-ordered-pipeline-parity.v1` receipt proving
exact dense output, sidecar, complete packed-v2 archive, and canonical order under
`scope: "production"`. The current cheap gate emits `scope: "synthetic_only"` and
`production_promotion_allowed: false`, so it cannot activate a production profile.

Build and select:

```sh
python3 tools/thread_profile_contract.py build \
  --receipt 72b-q3-8.json --receipt 72b-q3-12.json \
  --receipt 72b-q3-16.json --receipt 72b-q3-20.json \
  --expected-binary-sha256 "$BINARY_SHA" \
  --rss-limit-bytes "$RSS_LIMIT" \
  --output thread-profile.json

python3 tools/thread_profile_contract.py select \
  --profile thread-profile.json --tier 72B --rate q3 \
  --binary-sha256 "$BINARY_SHA"
```

The selector re-hashes and revalidates every bound receipt on every lookup. Wrappers
must consume its JSON output directly and remain fail-closed on exit status 2.

## Ordered bounded pipeline

The `ordered-pipeline` Cargo feature exposes `ordered_pipeline::run_ordered_pipeline`.
It overlaps three owners:

1. a source-read and deterministic-preprocessing producer;
2. the caller-owned encoder, which may invoke the block-parallel CPU encoder;
3. an ordered output sink.

Both boundaries use bounded synchronous channels. Every record carries a monotonically
increasing ordinal and conservative resident-byte accounting. `PipelineConfig` divides
each aggregate prepared/encoded budget across `depth + 2` possible resident records:
one active record at each adjacent stage plus the bounded queue. Order mismatches,
budget violations, stage errors, panics, and premature channel closure fail the entire
call. Block Viterbi scratch remains separately capped by `BlockParallelConfig`.
Caller-owned input descriptors are also outside the stage budgets, so production
adapters must pass lightweight path/range descriptors and load source bytes inside the
producer rather than retaining the corpus in the input vector.

The sink should always write a new temporary artifact and atomically promote it only
after the pipeline, exact-output comparison, and receipt finalization pass. A failed
sink can have written a prefix to its temporary file; the primitive does not claim
transactional rollback.

`quantize-model-ordered-pipeline` is the explicit container integration. With
`--ordered-pipeline-depth`, the source producer defers each selected tensor's f32 decode
and performs deterministic outlier/RHT preprocessing, the caller performs the existing
block-parallel encode, and the ordered sink admits `TensorResult` records in source
order. The existing dense, sidecar, STR1/STR2, SDSC/OUTL/SDSQ, and SPRV finalizers then
run unchanged. The path requires `--block-threads` and `STRAND_NO_GPU=1`.

The final STR2 bytes cannot be streamed concurrently: descriptor offsets, page layout,
optional side sections, and the outer SPRV seal require the complete tensor set. The
pipeline caps only prepared and encoded records in flight; the source archive bytes,
pass-through tensors, completed `TensorResult` records, finalizer memory, and separately
capped block scratch remain caller-owned. Identity-reuse mode is rejected and must use
the serial fallback.

`gate-quantize-model-ordered-pipeline` compares the explicit path with the same
block-parallel encoder run serially. Its cheap fixture enables RHT and proves exact dense
safetensors, JSON sidecar, complete STR2+SPRV archive, and canonical tensor order. It
emits `hawking.strand.quantize-model-ordered-pipeline-parity.v1` with synthetic-only
scope. `gate-ordered-pipeline` remains the lower-level stage/budget test.

## Current admission state

The implementation and synthetic exactness gate are ready, but no real tier/rate is
qualified by this work. Production qualification requires owner-free, source-bound
8/12/16/20 canaries with exact archive hashes, wall time, and peak RSS. Until those
receipts exist, wrappers must continue using their already reviewed execution path.

The explicit binary is wired but not promoted. Production remains fail-closed until an
owner-free, source-bound real-artifact run produces the end-to-end receipt above and the
8/12/16/20 tier/rate profile contract selects it. No runtime default changes here.
