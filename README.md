# Hawking

Hawking is an experimental systems stack for finding and running cheaper
physical representations of neural models. It pairs a Rust inference runtime on
Apple Silicon with a Python control plane that plans work, calls tools, runs
deterministic verifiers, and records evidence for every claim it makes.

The premise is that a model's cost is a property of its representation and its
execution, and that both can be searched. Hawking measures rather than asserts:
a result that has no receipt did not happen.

## What it does

- Runs GGUF and `.gravity` artifacts through a Metal execution path, with a
  CPU reference path for parity checking.
- Serves generation locally over an OpenAI-compatible HTTP surface.
- Searches quantized and packed representations, and evaluates them against
  measured quality rather than nominal bit width.
- Orchestrates long-running work through HCLI: work units, repair budgets,
  structured output contracts, and durable receipts.
- Keeps a resident model process alive across restarts, with a model-free
  supervisor that can inspect status without loading weights.

## Architecture

The detailed, current ownership map is [docs/CURRENT_ARCHITECTURE.md](docs/CURRENT_ARCHITECTURE.md).
In short, HCLI/AgentOS owns planning, typed tool calls, lifecycle, and
verification; the Hawking Rust crates own execution and serving. Receipts are
evidence, not an additional source of architecture.

## Current state

The runtime, the serving surface and the HCLI control plane are live and
tested. The representation search produces measured results; the strongest of
them are recorded under `research/`.

Work in progress, stated plainly because the distinction matters:

- HCLI self-improvement is running but has not yet landed an accepted change of
  its own authorship.
- Prompt throughput on the resident path is the current bottleneck. Prefill
  steps one token at a time, so it costs what decode costs.
- No Odyssey campaign has been run end to end, and no Odyssey wall time has
  ever been measured. Figures in the ledgers are budgets, not measurements.
- FPGA work is pre-board. There are no hardware results.
- The Event Horizon structural refactor does not alter live ascension state or
  perform physical optimization; it only consolidates source authorities and
  moves generated/operator state to owned roots.

## Build and run

```bash
cargo build --release
cargo run -p hawking -- generate --artifact <path> --prompt "hello"
cargo run -p hawking-serve

pip install -e .
hcli --help

# verification
python3 -m pytest
cargo check --workspace
```

Model weights are local inputs and are not part of this repository.

## Repository

| path | contents |
| --- | --- |
| `crates/` | Rust workspace: runtime, CLI, serving, benchmarks, and HCLI backend crates |
| `hcli/` | Python control plane, resident supervision, tools and verifiers |
| `tools/` | Campaign, evaluation and analysis tooling |
| `docs/` | Current architecture, control-plane boundaries, specifications |
| `research/` | Experiments, lab operators, archived work, evidence |
| `receipts/` | Acceptance and provenance records |
| `workspace/` | Local build and campaign working tree |
| `civilization/` | Roadmap and capability-graph state read by the control plane |

The visual HIDE/IDE surface is intentionally deferred. HCLI is the current
product surface; the visual client and its transport boundary will be rebuilt
against hardened VMCP rather than kept as a second active control plane.

## License

See [LICENSE](LICENSE).
