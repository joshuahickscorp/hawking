# Arsenal verification baseline — 2026-09-15

These are fresh local engineering observations for the reconciled tranche, not
provider latency or model-quality claims.

| Probe | Observation |
|---|---:|
| Selected relevant Python suite | 304 passed, 25 explicit exclusions, 19.64 s |
| Arsenal contract tests | 13 passed |
| `hawking-web` Rust crate | 2 passed |
| Rust content-aware repository search | 15 passed |
| Changed-module `py_compile` | passed |
| `git diff --check` | passed |
| Remote/provider spend | $0.00 |

The selected suite exercises typed binding, mutation intent/reconciliation,
completion evidence, scoped search, structural source adapters, WorkUnit
identity/packets, Auto planning/admission, Web projections, Goal/DAG
compatibility, remote gateway behavior, and result envelopes.

The separate live native-helper probe recorded 1 pass, 9 failures, and 4
deselected. All nine failures are the existing `processes.*` tests reaching a
helper binary that rejects the historical `processes` subcommand. The running
daemon and helper were not restarted to mask this mismatch.

Not measured in this source tranche: remote model latency/cost, cold/warm H-Web
reconnect, Rust-index IPC latency through a live adapter, full multi-worker Auto
execution, browser refresh/recovery, and daemon restart/resume. Those require a
disposable staging service and separate authority approval.
