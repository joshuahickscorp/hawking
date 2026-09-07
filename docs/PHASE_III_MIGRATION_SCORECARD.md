# Phase III migration scorecard

This branch is the structural refactor authority for `refactor/event-horizon`.
The live ascension worktree remains foreign and is not part of this scorecard.

## Product boundary

HCLI is the current headless product surface. The Rust `hcli` binary in
`crates/hide-backend` owns the consolidated HIDE backend: durable sessions,
backend protocol, tools, receipts, local-model composition, and runtime-facing
control. Python `hcli` remains the orchestration and resident skin where its
comparative advantage is the long-lived AgentOS loop and experimental provider
integration. Those are different ownership layers, not a second visual app.
Host-wide process observation and startup orphan reaping are now native in
`hide-backend::process_inspector`; Python exposes only a compatibility adapter
for the retained `/processes` and registry callers.

The desktop frontend, localhost `hide-serve` transport, and ACP server are
intentionally absent from this phase. They are recoverable from Git history and
may be rebuilt only after the external VMCP boundary is hardened. No current
visual authority is implied by this document.

## Phase IV minimum-operational steering

The Phase IV steer is now the active decision filter for this branch. The
ladder is `<1M` active LOC, then `<750K`, then `<500K`; if the remaining tree
still contains obvious semantic bloat, continue toward 400K/350K. Every
checkpoint reports both measures from `tools/loc/hawking_loc.py`:

* **Total active LOC** counts every tracked, non-archived, non-generated,
  non-vendored source line, including tests, research, shaders, and Markdown.
* **Minimum product LOC** counts only non-test HCLI Python, Rust/shader
  implementation under crate `src`/`shaders`, and the retained `tools/vmcp`
  boundary. It excludes research, acceptance/oracle harnesses, examples,
  documentation, and receipts.

The current minimum-product boundary is already below 500K, but total active
LOC is not: this is a valid product checkpoint and an open condensation target,
not a claim that the branch has reached the total-LOC floor. Repatriation must
follow profile → collapse → define minimum authority → implement Rust → switch
callers → verify → delete the Python owner → delete the bridge. Rust owns
durable machinery and computer/runtime state; Python remains for earned
cognition, research, and orchestration policy. VMCP remains an external,
independent boundary until hardened; no visual/IDE rebuild is in scope.

## Measured checkpoint

All LOC values come from `tools/loc/hawking_loc.py`; they are physical lines of
tracked executable source, including comments and blanks. The Phase III base is
`eec54c2ae`.

| checkpoint | active LOC | active files | Python | Rust | TypeScript | Rust code-only share |
|---|---:|---:|---:|---:|---:|---:|
| Phase III base `eec54c2ae` | 1,827,340 | 3,657 | 945,772 | 664,648 | 17,005 | 40.736% |
| after HCLI boundary deletion | 1,803,951 | 3,550 | 945,772 | 657,320 | 1,095 | 40.872% |
| after future-farm pruning | 1,609,847 | 3,273 | 751,583 | 657,320 | 1,095 | 46.485% |
| after Rust/headless compaction | 1,399,893 | 3,025 | 712,010 | 490,150 | 1,095 | 40.599% |
| current clean-tree census | 1,392,082 | 3,010 | 704,995 | 489,194 | 1,095 | 40.790% |
| current after restoring retained Qwen80 bridges | 1,385,526 | 2,999 | 694,193 | 493,431 | 1,095 | 41.370% |

The future farm deletion removed 278 uncalled Python files (192,748 physical
Python lines), leaving 60 tracked future-farm files at the Phase III
checkpoint: 57 retained Python
modules, the existing ledger and memory-traffic probe, and the retained test
helper shell script. The later Rust/headless compaction removed 171 uncalled
Rust example/shader files (168,974 physical lines) and 74 caller-free
headless Python files (39,566 physical lines). The current LOC measurement
includes this scorecard, which is now part of the tracked branch. Phase IV has
since promoted the VMCP/bench survivors and retired the dead headless Doctor
producers; the live namespace counts are recorded below. A follow-up reachability
pass removed the orphan Objective-C memory-traffic probe (641 active lines); its
historical receipt remains, and no current producer or acceptance path referenced
the source.

The Rust workspace census is now 18 packages, 12 binaries, and 73 examples.
The two retired resident-server entry points were `hide-headless` and the old
dirty-tier `research_server`; HCLI is the remaining HIDE backend binary.
The former one-consumer `hawking-index-query` package is now a binary and
Python-facts module owned by `hawking-index`; its command name and
`hawking.index.python_facts.v1` wire schema are unchanged.
The former library-only `hawking-bench` package is now the private
`hawking/src/bench` module owned by the `hawking` CLI; benchmark subcommands
and report schemas remain unchanged.

The branch must report the post-pruning measurement after the deletion commit;
historical receipts and generated graphs are not counted as source reduction.

The current row is the clean-tree measurement after the small ABI, contract,
receipt-path, closure-report, stream-render, runtime-identity, Rust-REPL,
duplicate-GoalIR, test-only-checkpoint, benchmark-package, and native-process
authority consolidations made after the historical compaction checkpoint. The
code-only
Rust share is computed as
`Rust / (Rust + Python + TypeScript + shell)`; it is not inflated by excluding
the deleted Rust examples or by treating Markdown as executable source.

Against the Phase III base, the current tree is down 435,258 active physical
LOC and 647 active files. Python is down 240,777 LOC; Rust is down 175,454 LOC
because the deletion wave removed uncalled historical examples rather than
restoring them to improve a ratio. The code-only Rust share therefore moved
from the historical 40.736% checkpoint to 40.790%; the native process
authority and Python-side deletions recover some Rust share, but this remains
an open migration target rather than a claimed success.

The tracked tree is 10,329 files / 461,220,117 bytes versus the Phase III base
of 11,058 files / 479,552,373 bytes: down 729 files and 18,332,256 bytes.

## Closure status

This scorecard is an active checkpoint, not a declaration that every Phase III
gate is closed.

| gate | current evidence | status |
|---|---|---|
| at least 10,000 active source LOC removed | base 1,827,340 -> current 1,392,082 | met |
| minimum product LOC below 500K | `product_LOC=498,094`, `product_files=635` | met; conservative boundary |
| total active LOC below 500K | `total_LOC=1,392,082` | open; Phase IV ladder in progress |
| Rust active share materially increased | historical 40.736% -> current 40.790% after native process/VMCP boundary migration | open |
| Rust workspace check | `cargo check --workspace` | pass |
| relevant Rust tests | `cargo test --workspace --lib --bins` | pass |
| full Python HCLI suite | 1,522 passed, 49 protected/evidence failures, 7 skipped (1,578 collected) | open; no failures hidden; seven moved process-policy assertions now run in Rust |
| no new unexplained failures | focused regressions closed; missing live sovereign receipts remain | open |
| examples and launch surface reduced | 235 -> 73 examples; 14 -> 12 binaries | met |
| remaining Rust packages have an ownership boundary | 18 packages after folding two one-consumer facades | met |
| clean working tree | `git status --short` empty | met |

The 49 HCLI failures are not converted into green receipts: 48 require ten
missing live sovereign receipts (G002-G008, G012-G013, G015), and one existing
G014 negative-science receipt still reports a non-zero dead-family duration.
Those are evidence prerequisites for a later closure run, not source-code
failures that this branch may fabricate away.

## Candidate census and migration score

The retained-Python census uses physical `wc -l` counts, `rg` import/call-site
counts, the production command paths, and the retained test suite. Frequency and
CPU wall are classified from those call paths (hot request path, resident loop,
or CLI/evidence path); they are not presented as a profiler measurement. The
priority score is a 0-10 triage score: two points each for durable/control-plane
responsibility, process or concurrency ownership, hot-path/CPU exposure, a
compatible Rust owner, and removable duplicate authority. It ranks work; it
does not authorize a port without schema and restart parity.

| rank | retained module | LOC | callers / frequency | state, process, IO, or encoding load | Rust/authority disposition | score |
|---:|---|---:|---|---|---|---:|
| 1 | `hcli/agentos/resident.py` | 2,437 | controller, AgentOS runtime, resident CLI; resident-loop hot path | process lifecycle, concurrency, durable mission state, JSON | next Rust candidate after resident parity and restart/receipt evidence | 8 |
| 2 | `hcli/controller.py` | 2,028 | commands, runtime, tools, resident; hot request/control path | stateful orchestration, filesystem writes, context/session serialization | retain as Python orchestration until a Rust turn owner is proven | 6 |
| 3 | `hcli/mission.py` | 1,588 | controller, resident, status/steering; resident hot path | mission/work-unit transitions, scheduler coordination, JSON state | retain; Rust scheduler is not the same mission schema | 6 |
| 4 | `hcli/runtime.py` | 1,394 | controller, AgentOS runtime, native bridge; hot model path | backend selection, subprocess/runtime lifecycle, resource gates | retain behind Rust runtime-serving boundary until parity | 6 |
| 5 | `hcli/resources.py` | 1,156 | controller, runtime, resident; hot admission path | RAM/GPU limits, mutation lock, process health, durable diagnostics | retain; overlaps several Rust policies but has no drop-in schema owner | 6 |
| 6 | `hcli/engine.py` | 6,105 | command/controller/runtime/tool paths; hot execution path | prompt/context state, tool loop, provider calls, serialization | retain; highest LOC but no safe one-to-one Rust owner yet | 4 |
| 7 | `hcli/backends.py` | 2,436 | runtime/engine/provider paths; model-call hot path | provider/network/process integration and response encoding | retain for comparative provider/hardware advantage | 4 |
| 8 | `hcli/tool_registry.py` | 2,406 | engine, commands, resident; every tool turn | registry dispatch, effect policy, filesystem/process IO, JSON | Rust HIDE registry owns backend tools; Python AgentOS registry remains distinct | 4 |
| 9 | `hcli/workunit.py` | 626 | mission/controller/steering; frequent state transitions | durable identity and JSON; SHA-256 schema differs from Rust BLAKE3 objects | retain until an explicit compatibility migration is specified | 2 |
| 10 | `hcli/knowledge.py` | 524 | controller, AgentOS runtime, tool registry; context turns | workspace hot index plus gzip cold archive and JSON | retain; not interchangeable with `hawking-context` classed SQLite memory | 2 |
| 11 | `hcli/session.py` | 264 | controller resume/compaction; turn lifecycle | transcript hot tail, semantic checkpoint, gzip history, legacy JSON | retain; Rust `SessionRegistry` owns identity/lineage, not this transcript schema | 2 |
| 12 | `hcli/processes.py` | 205 | commands and tool registry; operator/diagnostic path | host `ps`, footprint/RSS, argv classification, SIGTERM | consolidated: 7 policy assertions moved to Rust; Python is a compatibility skin | 10 |

The high-scoring process surface is complete because the native owner now
passes live classification, workspace claims, safe orphan filtering, RSS
fallback, dry-run reaping, and restart-oriented HCLI smoke. The remaining
high-value rows are intentionally retained: their Rust neighbors differ in
identity, schema, or authority semantics, so deleting Python now would create a
dual-translation layer rather than consolidation. The next real migration
target is the resident/mission boundary, gated by a Rust owner that can replay
the same work-unit identity, repair budget, receipts, and restart behavior.

## Phase IV namespace dispositions

The first namespace wave applies the explicit Phase IV rule that `future` and
`headless` are not architectural kingdoms. The live VMCP disposition, VMCP
capability/oracle probes, HCLI VMCP integration, and structured-output probe
were promoted to `tools/vmcp` or `tools/bench`. The uncalled headless Doctor
producer and the nested benchmark driver were deleted; their receipts remain
historical evidence. The follow-up reachability pass also deleted one orphan
Objective-C probe, leaving `tools/future` at 55 tracked files; `tools/headless`
has 54. Remaining files are not silently treated as product:
they are the next PROMOTE/MERGE/ORACLE/RESEARCH/DELETE queue, ranked by active
LOC, callers, authority, and evidence coverage. A second pass then removed
three uncalled future producers (`resident_supervisor`, `phase_listeners`, and
`detached_trial`, 5,374 Python lines) after changing the lifecycle audit to
receipt-only for their historical trials; `RESOURCE_AVAILABLE` now routes to
the existing profile handler rather than a second future supervisor.

## Build and test performance evidence

Measured on 2026-09-04 in this worktree with the configured external Cargo
target directory. Clean measurements used isolated temporary target directories;
warm measurements reused the configured target. These are wall-clock baselines,
not claims about optimal parallelism.

| workload | mode | wall time | result |
|---|---|---:|---|
| `cargo check --workspace` | clean | 18.47s | pass |
| `cargo check --workspace` | warm | 8.93s | pass |
| `cargo test --workspace --lib --bins` | clean build + test | 62.84s | pass |
| `cargo test --workspace --lib --bins` | warm build + test | 23.64s | pass |
| `python3 -m pytest hcli --collect-only -q -p no:cacheprovider` | collection | 1.54s | 1,576 selected / 1,578 discovered |
| `python3 -m pytest hcli -q -o addopts='' --tb=line` | full retained suite | 69.00s | 1,522 pass / 49 protected-evidence failures / 7 skip |

The next performance pass should isolate Cargo linker/codegen/dependency walls,
measure `-j` choices, and compare pytest worker counts and shared-fixture
contention. Those measurements remain open; no unstable parallelism or cache
reuse is being claimed by this scorecard.

## Python-to-Rust migration register

The register is deliberately narrow. A Python module is not a migration target
just because Rust can express it; it must have a durable/runtime/control-plane
role, a measured caller, and a clear Rust owner.

| concern | current authority | Rust owner | disposition |
|---|---|---|---|
| backend command ingress and JSONL control | `crates/hide-backend/src/bin/hcli.rs` + `hcli_bridge` | `hide-backend` / `hide-protocol` | consolidated; visual clients deferred behind VMCP |
| durable HIDE sessions and receipts | `hide-backend::services`, `replay`, `rewind` | `hide-backend` | Rust authority; no Python duplicate for this surface |
| local model/runtime serving | `hawking-serve` and `hide-backend::supervisor` | Hawking runtime crates + `hide-backend` | Rust authority; Python may supervise its resident body |
| host process observation and startup reaping | former `hcli.processes` implementation | `hide-backend::process_inspector` via `hcli processes` | consolidated; Python is a thin compatibility adapter and model tools remain read-only |
| AgentOS resident loop and comparative provider integrations | `hcli.agentos`, `hcli.runtime`, provider modules | none yet | retained in Python until a parity-tested Rust owner exists |
| ordinary crash-safe Python writes | `hcli.persist` | none yet | retained; narrow, tested Python advantage, not a second HIDE store |
| VMCP perception | external `visionmcp` plus `tools/vmcp` adapters | external VMCP boundary | retained as a boundary; visual rebuild waits for hardening |
| one-shot research and retired experiment runners | `tools/future` | none | delete uncalled producers; retain receipts and live call-path modules |

## Bridge register

| bridge | owner | reason it exists now | removal condition |
|---|---|---|---|
| Python AgentOS -> resident/native provider | Python `hcli.agentos` | long-lived resident and hardware-specific provider work is not yet parity-ported | a Rust resident loop passes restart, receipt, and negative-control parity |
| Python process skin -> native HCLI | `hcli.processes` -> `hcli processes --json` | preserves legacy REPL/tool result shape while Rust owns host inspection and signals | Python process callers migrate directly once the native binary is the installed entry point |
| HCLI backend -> `hawking serve` | Rust `hide-backend::supervisor` over HTTP | keeps the engine crate boundary narrow while the runtime remains independently served | a stable hardened VMCP/runtime transport replaces the compatibility launcher |
| VMCP adapter -> external `visionmcp` | `hcli.vmcp_adapter` / `tools/vmcp` | VMCP is an external package and must remain an independent sensory authority | hardened VMCP contract covers visual/IDE clients and evidence semantics |

No bridge is allowed to grow a second scheduler, durable state store, or
semantic API. A future Rust port must delete or repatriate its Python owner in
the same change that moves the callers.

## Future-farm pruning rule

The retained `tools/future` set is the transitive closure of current production
imports plus the explicit lifecycle and FPGA acceptance call paths. The removed
files had no production import/call path outside the future farm. They are not
silently reclassified as documentation: their executable source is removed and
their historical receipts remain immutable. Future test commands must target a
retained owner or a current acceptance test; there is no separate uncalled
future test farm in the product build.
