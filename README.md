# Hawking

Hawking is a local model-engineering system for Apple Silicon. It combines a
native Rust/Metal execution stack with a durable local operator that can plan
work, call bounded tools, verify results, and preserve the receipts and state
needed to resume after an interactive session ends.

The project treats representation, execution, and evidence as separate
problems. A compact artifact is not assumed fast, a successful command is not
assumed correct, and a result without an inspectable receipt is not treated as
an accepted result.

## What is active today

- Native loading and execution of GGUF and `.gravity` artifacts, with Metal
  paths and CPU-reference parity work.
- Local generation and serving over an OpenAI-compatible HTTP surface.
- Representation and qualification work for packed and quantized artifacts,
  evaluated against measured quality rather than a nominal bit width alone.
- The `hawking` operator: interactive work, typed tool calls, structured
  contracts, checkpoints, repair budgets, receipts, and a persistent goal
  bank.
- A resident supervisor that owns long-lived missions and model lifecycle;
  the resident can be inspected or restarted without loading weights just to
  answer a status question.
- Local web, report, build, serving, and recovery surfaces built on the same
  durable state rather than on a separate UI control plane.

## Run it locally

Hawking has a Rust execution layer and a Python operator layer. Build the
runtime, install the local operator, then use the command from the repository
or an editable environment:

```bash
cargo build --release

python3 -m pip install -e .
hawking

# Long-lived local operation
hawking resident start
hawking resident status
hawking web
```

`hawking` opens the interactive operator; it also accepts one-shot natural
language work. The resident stores durable local state under `.hawking/`,
including missions, checkpoints, background work, goal-bank state, receipts,
and recovered failure context. `hawking --help` is the authority for the
available operator commands.

Model weights are local inputs and are not included in this repository.

## Architecture

The current ownership map is in
[docs/CURRENT_ARCHITECTURE.md](docs/CURRENT_ARCHITECTURE.md), with an
operator-oriented guide in [docs/HAWKING_OPERATOR.md](docs/HAWKING_OPERATOR.md).
At a glance:

```text
interactive CLI / web / resident
             │
  missions · goal bank · tools · checkpoints · receipts
             │
Rust runtime · serving · artifact qualification · Metal execution
             │
            local artifacts and measured evidence
```

The Python `hawking/` package provides orchestration, residency, tool and
verification policy. The Rust workspace owns runtime execution, serving,
artifact handling, benchmarks, and durable backend authority. The visual
HIDE/IDE surface remains deliberately deferred behind Hawking's hardened
perception boundary; the CLI, web, and operator surfaces are the active
product.

## Evidence and current limits

Hawking has functioning runtime, serving, and durable-operator surfaces. Its
representation work includes measured artifact results under `research/`, and
the operator persists and recovers work as a first-class capability.

Important boundaries remain explicit:

- The self-improvement loop has not yet landed an accepted change of its own
  authorship.
- Resident-path prompt throughput is still constrained by prefill behavior;
  individual kernel gains are not presented as end-to-end throughput proof.
- No Odyssey campaign has run end to end, and its budget figures are not wall
  time measurements.
- FPGA work is pre-board; there are no hardware results.
- Structural work around Event Horizon consolidates authorities and state
  ownership. It does not by itself perform physical optimization.

## Verification

```bash
python3 -m pytest
cargo check --workspace

# Complete Rust source/compile lanes; neither runs model or hardware cases.
tools/ci/rust_test_fast.sh check
tools/ci/rust_test_fast.sh compile

# Focused execution examples.
cargo test -p hawking-core --test q4k_fast_parity
tools/ci/rust_test_fast.sh policy
```

The fast lane compiles the integration source surface through checked aggregate
targets while preserving explicit isolated targets for focused runtime work.
Use a named target when executing a concrete integration case rather than
assuming that compilation is equivalent to a model or hardware result.

## Repository map

| Path | Contents |
| --- | --- |
| `crates/` | Rust runtime, serving, CLI, benchmarks, and backend crates |
| `hawking/` | Python operator, residency, tools, verifiers, and web surface |
| `tools/` | Campaign, evaluation, and analysis tooling |
| `docs/` | Current architecture, operator guides, and specifications |
| `research/` | Experiments, lab operators, archived work, and evidence |
| `receipts/` | Acceptance and provenance records |
| `workspace/` | Local build and campaign working tree |
| `civilization/` | Capability graph and roadmap state used by the operator |

## License

See [LICENSE](LICENSE).
