# Hawking

Hawking is a Rust inference and serving stack for Apple Silicon. It loads
supported GGUF and `.gravity` artifacts, provides CPU/reference and Metal
execution paths, and exposes local generation through a CLI and HTTP server.
The Python HCLI package adds a provider-neutral control surface, structured
results, resident-worker controls, and evidence capture.

This repository contains experimental model runtimes as well as the reusable
inference and serving substrate. Model weights are local inputs; they are not
part of the repository.

## What runs here

- `hawking-core` owns artifact loading, runtime state, attention/decode,
  quantized and packed matmul, KV-cache handling, batching, and Metal kernels.
- `hawking` provides the command-line surface for generation, tokenization,
  serving, Gravity artifact operations, benchmarking, and related checks.
- `hawking-serve` exposes the local HTTP surface, including
  `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
  `/v1/hawking/generate`, `/v1/hawking/tokens`, `/v1/models`, `/healthz`, and
  `/metrics`.
- HCLI accepts local/native or OpenAI-compatible providers and records the
  selected provider, artifact, tokenizer, runtime, capability, and verification
  context in structured results.
- The HCLI resident path can run a model-free supervisor with disposable
  workers, bounded restart/continuity state, and explicit child-job ownership;
  status inspection does not open model weights.

The workspace also includes HIDE support crates, research/evaluation tools,
and application scaffolding. These are workspace members, but not every member
is part of the default build or a production-ready adapter.

## Architecture

```text
model artifact -> hawking-core -> hawking CLI / hawking-serve
                                  \
                                   -> HCLI provider and resident controls
```

The Rust workspace is defined in `Cargo.toml`. The Python package and console
scripts are defined in `pyproject.toml`. Receipts and benchmark reports record
the conditions and limitations of measurements; an isolated kernel or organ
benchmark is not automatically a full-token generation result.

## Build and run

From the repository root:

```bash
cargo check -p hawking
cargo test -p hawking-core --lib

cargo run --release -p hawking -- generate \
  --weights /path/to/model.gguf \
  --prompt "Write a function that reverses a string."

cargo run --release -p hawking -- serve \
  --weights /path/to/model.gguf

cargo run --release -p hawking -- gravity serve \
  --artifact /path/to/artifact
```

Install the Python/HCLI surface when needed:

```bash
python3 -m pip install -e .
hcli --help
```

Metal execution requires Apple Silicon and the relevant local runtime inputs.
Use `hawking --help`, `hawking serve --help`, and `hawking gravity --help` for
the complete command-specific options.

## Evidence-bound performance notes

The [full-sequence parallelism report](evidence/parallelism/FULLSEQ_CAPTURE_PARALLELISM_FINDINGS.json)
records a synthetic resident-weight capture/export fixture: two workers reached
1.398× and four workers 1.788× the serial receipt baseline, with byte-identical
merged outputs. This is evidence for sequence-sharded capture on that fixture,
not a universal inference-throughput claim.

The [GLM teacher-forced report](evidence/parallelism/GLM_TEACHER_FORCED_PARALLELISM_FINDINGS.json)
records a bit-exact safe sequence-shard path and explicitly does not claim a
full-scale speedup. Link-bound or model-specific results must be measured again
under their own workload and hardware conditions.

## Repository layout

- `crates/` — Rust inference, serving, context, evaluation, and support crates.
- `hcli/`, `lab/`, and `tools/` — Python control, experiments, verification,
  and maintenance tooling.
- `docs/` — current architecture and archived state/reference documents.
- `evidence/` and `receipts/` — measurements, classifications, and sealed
  provenance records.
- `app/` — frontend and desktop scaffolding.

The HTTP compatibility surface is intentionally narrower than every provider
API; for example, `/v1/responses` and `/v1/messages` are not implemented by
`hawking-serve`.

## Accelerator platform loop

The platform steer is consolidated in the canonical [H-ROADMAP](/Users/scammermike/Downloads/H-ROADMAP.md)
as the C01–C64 graph plus the P01–P18 compounding queue. The compact,
receipt-derived scoreboard can be regenerated with:

```bash
/usr/local/bin/python3 tools/accelerator/scoreboard.py
```

It records model/backend/representation and physical cost fields while keeping
unmeasured values unknown. The Physical Graph promotion rule still requires
measured complete useful work, independent capability evidence, and a protected
benchmark. The ANE lane remains public-Core-ML-only; CUDA is portability/HWIR
work on this Apple host, not a CUDA-hardware performance claim.

The Flash continuation now has a reusable `TerminalExecutor` that keeps the
source index, terminal weights, Metal context, pipelines and workspaces alive
for a native session. Its build boundary is recorded in
[FLASH_TERMINAL_EXECUTOR_COMPILE.json](/Users/scammermike/Downloads/hawking/receipts/headless/FLASH_TERMINAL_EXECUTOR_COMPILE.json);
physical accepted-token and protected performance claims remain gated on the
next native run.
