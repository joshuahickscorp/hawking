# aud18 — schema and API surface

Discovery/audit only. HEAD `04193ccbc`. Evidence is `SOURCE_INSPECTION` / `STATIC_VERIFICATION` via `git show HEAD` / `git grep HEAD` (this worktree is sparse; `hcli/` is in git but not on disk). Nothing was executed on Metal/ANE/FPGA. No claim is `MEASURED`.

## Answer

The repository does not have one schema family. It has **several real, called systems** (WorkUnit/Mission, lab seal, RuntimeProfile, hide WriteLease, BackgroundJob) sitting next to **a thousand one-shot receipt schema strings** and **569 environment variables**, 393 of them experimental. The same English words — receipt, experiment, genome, obligation, law, mission, profile, lease — name incompatible types.

What is actually shared today is not a god-object. It is `lab.receipts.seal` / `verify` (186 non-test importers). The typed `lab.receipts.Receipt` class is almost unused (two production constructors, four on-disk `hawking.lab.receipt.v1` files).

**Do not merge these into one universal schema.** Reuse small primitives (id, RFC3339 time, content hash, parent ref, lifecycle vs verdict, evidence ref, seal). Keep domain constraints.

## Concept table

Definitions without callers are not counted as `CALLABLE`. Imports are not call sites. Counts are `Symbol(` construction / method use.

| Concept | Classification | What exists | Callers of the symbol | Hole |
|---|---|---|---|---|
| **WorkUnit** | `INTEGRATED` | `hcli/workunit.py:33` dataclass; DAG in `goal.py` / `dag_store.py` | 74 production `WorkUnit(` + Mission/Scheduler/Executor | `to_dict` drops `tool` / `tool_arguments` |
| **ResultEnvelope** | `INTEGRATED` | `hcli.agentos.result.v1` | `engine.py:4797`, `agentos/runtime.py:767`, `escalation.py:219` | Competing envelopes: `HCLI_RESULT_SCHEMA` (constrained decoding), `hawking.events.canonical.v1` |
| **Receipt** | `INTEGRATED` (as *seal*, not as one type) | `hawking.lab.receipt.v1`, `GateEvidence`, `ProviderReceipt`, 1140 on-disk schema ids | Typed `Receipt(` only `lab/runtime.py:72,364`. `seal()` is the real API | Word “receipt” is not a type |
| **Experiment** | `INTEGRATED` (`lab.spec`); `SCAFFOLDED` (`ExperimentRun`) | Two `ExperimentSpec` classes + `AcceleratorExperimentSpec` + unused Rust `ExperimentRun` | `lab/runtime.py` loads `hawking.lab.experiment_spec.v1` | `hawking.lab.experiment.v1` is a different object with the same class name |
| **Benchmark result** | `CALLABLE` | `BenchReport` (Rust), `classify_window` dict, `lab/.../BenchResult` | `hawking bench` wired; no shared `BenchmarkResult` | Physical numbers live as one-shot receipts. GPU claims `BLOCKED_AUTHORITY` |
| **Mission state** | `INTEGRATED` | Python `Mission` → `.hcli/mission/state.json`; Rust `haider.mission.v1` → `.haider/mission.json` | 56 production `Mission(` | Two durable documents, two paths |
| **GoalIR** | `TESTED` | `GoalNode` frozen dataclass + tokenizer/graph/compile adapter | Production `GoalNode(` only in `goal_tokenizer.py:537`. Mission uses `GoalCompiler` → WorkUnits | Adapter `goal_compile.schedule` is test-only |
| **Law** | `CALLABLE` | Two incompatible `class Law` in `tools/future/` | Construction in odyssey2 store / autonomy_scars / propagate. Acceptance `scars()` is synthetic | Not on hcli Mission path. Not `ADVERSARIALLY_VERIFIED` |
| **Scar** | `CALLABLE` | Two `class Scar` in `tools/future/` | Same cluster; CAPABILITY_GRAPH: registry, not a fresh law attack | Would be `ABSENT` if you only grepped `hcli/` + `crates/` |
| **Hardware profile** | `CALLABLE` (named type `ABSENT`) | `MachineProbe`, `MachineGenome` (two producers), `DeviceLimits` | `resolve_runtime_limits`, `hcli max` | Genome is a prior, not live truth. FPGA genome `BLOCKED_HARDWARE` |
| **Model profile** | `INTEGRATED` | `RuntimeProfile` enum, `KernelProfile`, `ResidentProfile`, two `RuntimeGenome` | clap `--profile` + `lever_plan()` | Profile covers ~5 of 72 `HAWKING_QWEN_*` vars |
| **Resource lease** | `INTEGRATED` (named type `ABSENT`) | `WriteLease`, `PortLease`, `WorktreeLease`, `SingletonLease`, `FixtureHeavyLease`, `MutationLock` | hide tools, hide-fleet isolate, lab runtime, WorkUnit `assign_ready` | Four lease systems, not one |
| **Background job** | `INTEGRATED` | `hcli.agentos.background_job.v1` | Store + `hcli agentos background *` + resident + accelerator runner | Distinct from hide `JobRecord` |

## Overlapping metadata

These fields are redefined, not reused:

- **id** — `WorkUnit.id`, `GoalNode.id` (UPPER_SNAKE), `Obligation` `G001`, `job_id`, `campaign_id`, `law_id`/`identity`, `lease_id`, ULID event ids
- **time** — ISO-Z (`Receipt.at`), epoch float (Mission/WorkUnit/BackgroundJob), `created_ms` (hide/haider)
- **hash** — `content_identity`, `content_signature`, `seal_sha256`, `SourceRef.sha256`, kernel layout/shader hashes
- **status** — at least eight enums (WorkUnit 6, GoalNode 6, Obligation, Mission.phase, AgentState, BackgroundJob 7, catalog ExperimentSpec, ResultEnvelope verdict, benchmark_class)
- **parent** — dependencies vs repairs vs `parent_ultragoal` vs `parent_job_id`
- **provenance** — GoalNode utterance-lineage enum vs hide-core trust-level `Provenance` (same word, different axis)

## Shared primitives (not a god-schema)

Already in production and worth doubling down on:

1. **`lab.receipts.seal` / `verify`** — the only integrity primitive with broad callers.

Propose, as *small* shared types, not one object:

| Primitive | Shape | Reuse | Do not |
|---|---|---|---|
| ObjectId | string + prefix `wu_`/`goal_`/`job_` | identity fields | a global registry |
| UtcTimestamp | RFC3339 Z in JSON | stop mixing ISO-Z / epoch / ms | force hide wire off ms |
| ContentHash | `{alg, hex}` | identity hashes and seals (keep the *role* distinct) | one hash field for both identity and integrity |
| ParentRef | `{kind, id}` | repairs, ultragoal, parent job | collapse DAGs into a single parent |
| LifecycleStatus | pending…cancelled | WorkUnit + BackgroundJob + Mission.phase only | merge with GoalNode or Obligation |
| Verdict | ACCEPT\|BLOCKED\|UNVERIFIED | ResultEnvelope | treat verdict as lifecycle |
| EvidenceRef | `{path, sha256, kind}` | GateEvidence, SourceRef, receipt_paths | inline blobs |
| ErrorRecord | `{type, message, traceback?}` | engine receipts, failures[], job.error | |
| ProvenanceStamp | GoalNode’s kind + EvidenceRef[] | utterance lineage | merge with hide-core trust Provenance |

Keep domain-specific: WorkUnit repair budget and typed-tool fields (and **persist them**), GoalNode promotion guard, GateEvidence three-role split, KernelProfile hashes, Law scope lattice, RuntimeProfile lever_plan with env override.

## Receipt compression

| Metric | Value | How |
|---|---|---|
| Unique `"schema"` values under `receipts/` | **1140** | `git grep '"schema":' HEAD -- receipts/` |
| Hits of those fields | 9928 | same |
| Unique Python `SCHEMA = "` (non-test) | **1062** | `git grep`; `lab/operators` alone 532 lines |
| Unique hcli `SCHEMA` constants | **76** | product-ish plus Flash-Next science gates |
| Typed `hawking.lab.receipt.v1` files | **4** | `git grep -l` |
| Typed `Receipt(` production sites | **2** | `lab/runtime.py` |
| `from lab.receipts import` non-test files | **186** | almost all `seal`/`verify` |

Top on-disk schema ids (instances, not types):

| n | schema |
|---|---|
| 1938 | `hawking.gravity.activation_weighted_svd_low_rank.v1` |
| 1198 | `hcli.agentos.background_job.v1` |
| 568 | `hcli.agentos.flash_noetic_component_body.v1` |
| 548 | `hawking.audit.wake_condition.v1` |
| 488 | `hawking.flash_noetic_q4_kernel_parity.v1` |

### Migration map (preserve everything)

Never delete old receipts. They are provenance.

| From | To | Still produced? | Still read? |
|---|---|---|---|
| One-shot `hawking.ascension.*` / `hawking.gravity.*` / `hawking.flash_*` | Keep files. New producers: domain payload + `{schema,id,at}` + `seal()` | Yes, continuously | Yes — `civilization/build_state.py` and each operator’s own `RESULT_SCHEMA` |
| `hawking.lab.receipt.v1` | Canonical *lab campaign* receipt | Yes (`lab/runtime.py`) | Yes, but only 4 files |
| `hawking.lab.gate_evidence.v1` | Canonical gate identity | `write_gate_evidence` | `lab/rules.py` |
| `hcli.agentos.result.v1` | Canonical AgentOS/Engine result | Engine + AgentOS + escalation | knowledge.py (dict fields) |
| `hcli.agentos.background_job.v1` | Canonical background job | `BackgroundJobStore._write` | inspect/list/resume |
| `hcli.provider.receipt.v1` | Provider-call receipt (not a campaign) | `executors.py` | provider path |
| `hawking.events.canonical.v1` | Event envelope, not a result | hawking-events | hide-core |
| `hawking.lab.experiment.v1` vs `….experiment_spec.v1` | **Do not merge** — two ExperimentSpec classes | both loaders | both |
| `hcli.provider.runtime_genome.v1` vs `hawking.headless.runtime_genome.v1` | Rename one class | both | both |
| `hawking.cli.surface.v1` JSON | Regenerate from clap, or stop claiming sole authority | adapters codegen | completion/help; **stale** |
| Engine `.hcli/receipts/*.json` | Optionally annotate; do not rewrite | engine | `/receipts` |

## API surface counts

### CLI verbs

| Surface | Count | Notes |
|---|---|---|
| `hawking` clap leaves | **23** | serve, gravity{plan,condense,serve,execute,verify}, generate, tokenize, bench, autotune, bench-q4k-shapes, doctor, profile-rank, stats, version, batch-hash, shader-hash, verify, bake-sidecar (**stub**), condense, press, fit |
| Generated `HAWKING_CLI_SURFACE.json` | **5** | serve, generate, **adapters** (not in clap), doctor, version — `STALE_ROADMAP_TEXT` |
| Python `hcli` top-level | ~12 | install-shims, delegate run/status/steer/result/abort, connectivity, flash-next, agentos, resident/daemon, positional |
| Python slash commands | **21** + 3 aliases | `hcli/command_registry.py COMMANDS` — wired via `_cmd_*` |
| `hcli agentos` | **45** top-level + nested resident/background | 59 `add_parser` calls; science gates mixed with control plane |
| Rust `hcli` bin | ~16 argv verbs | **different product** |
| haider REPL | ~24 slash verbs | hide-backend haider |

Accidental growth: four names for gravity plan (`condense` / `press` / `gravity condense` / `gravity plan`); agentos Flash-Next gates as CLI verbs; two binaries named `hcli`.

### Tools

| Registry | Distinct names |
|---|---|
| HIDE builtin (`tooling_registry.rs`) | **23** (`fs.*`, `edit.*`, `shell.*`, `git.*`, `memory`, …) |
| AgentOS `ToolSpec(` positional | **50** (plus aliases `filesystem.*`, `receipt.inspect`, three git-safe refusals) |
| hide-protocol command catalog | **57** intents |
| HAIDER ToolBus | ~26 (subagent; `fs.read`/`git.status` overlap HIDE names) |

### Python public (`^def` / `^class`, non-test, git show)

| Area | Modules | Public defs | Public classes |
|---|---|---|---|
| hcli | 118 | 423 | 176 |
| lab | 311 | 1729 | 595 |
| tools/future | 262 | **4453** | 754 |
| hawking-experiments | 148 | 1822 | 94 |
| tools/accelerator | 38 | 274 | 128 |
| civilization | 3 | 12 | 0 |

`tools/future` is larger than hcli+lab combined. That is where Law/Scar live. Public-def count is **not** reachability.

### Rust public (non-test, non-example)

| | pub struct | pub enum | pub fn | pub trait | pub use |
|---|---|---|---|---|---|
| **All 23 crates** | 1757 | 471 | 1107 | 86 | 294 |
| hawking-core alone | 655 | 133 | 751 | 22 | 50 |

Accidental growth center: `hawking-core` public `gravity_deepseek_v4_*` modules.

### Environment variables

569 unique names from `os.environ` / `getenv` / `env::var` / `option_env!` / `set_var` in `*.py` `*.rs` (excluding `receipts/` and `hawking-experiments/`).

| Class | Count | Rule of thumb |
|---|---|---|
| CANONICAL | 110 | paths, bind, model identity, credentials, stdlib thread pools, `HAWKING_GRAVITY`, `HAWKING_HOME`, `HIDE_*`, core `HCLI_MODEL_*` |
| DEBUG | 65 | TRACE / CAPTURE / TIMING / SKIP_ / FORCE_ / PROBE — **keep for bisection** |
| EXPERIMENTAL | 393 | family fuse levers, `NOETIC_*`, `GLM52_*`, most `HAWKING_QWEN_*`, `N0xx_*` |
| DEPRECATED | 1 | `HAWKING_SPEC_DECODE` (eagle5 commented removed) |

Prefixes: **220** `HAWKING_*`, **72** `HAWKING_QWEN*`, **59** `HCLI_*`, **13** `HIDE_*`, **100** names containing `QWEN`. 291 experimental names appear in a single file.

`RuntimeProfile` already maps a bundle (`HAWKING_QWEN_Q4K_LMHEAD`, `Q4K_PREDEC`, `PREDEC_F16SCALES`, `VOCAB_PRUNE`, `FFN_DOWN_Q4K`, plus `HAWKING_ENERGY_EFFICIENT` on Efficient). Explicit env still wins. **Do not delete the rest**; classify them as profile overrides.

Loud: `HAWKING_ENERGY_EFFICIENT` is **set** by Efficient and **never read** (`git grep` has no `getenv`/`env_on`). Energy mode is applied via `--energy-mode` / Apple Fit, not this var.

Full classified list: `aud18-schema-and-api-surface.json` → `api_surface.environment_variables.vars`.

### Configuration keys

- Python: `~/.config/hcli/config.json` and `{workspace}/.hcli/config.json` — `model`, `enable_thinking`, `response_schema`, `model_tokens` (env wins).
- hide-core `HideConfig`: runtime / persistence / security / context / index nested keys.
- Durable state paths: `.hcli/mission/state.json`, `.haider/mission.json` (different schemas).

### Schema names (generated hawking.*)

Ten versioned ids in `HAWKING_SCHEMA_MIGRATIONS.json` (registry, ABI, events, bridge, **cli.surface**, artifacts, profiles, runtime_capabilities, fabric.placement, tool_effects). CLI surface document is the stale one above.

## Surprises (what would settle them)

1. **Two `hcli` binaries.** Settle: rename one; confirm which the live launchd daemon execs (read-only `ps`/`PYTHONPATH`, do not signal).
2. **Generated CLI surface is not the CLI.** Settle: regenerate from clap or drop the “sole surface” claim. `adapters` is advertised and not implemented.
3. **Typed `Receipt` is not the receipt API; `seal()` is.** Settle: mutation-check deleting `class Receipt`.
4. **GoalIR is tested and unused by Mission.** Settle: one production `goal_compile.schedule` call, or label it compiler-only IR.
5. **`HAWKING_ENERGY_EFFICIENT` is a dead write.** Settle: `env_on` at EnergyMode resolve, or stop setting it.
6. **569 env vars, 72 Qwen levers, 5 in the profile.** Settle: remaining levers belong in KernelProfile JSON with env override.
7. **Duplicate type names** (RuntimeGenome, Obligation, ExperimentSpec, Law, Scar, MachineGenome, ResourceLimits). Settle: rename, do not merge.
8. **`tools/future` is a second codebase.** Settle: product vs scratch. Acceptance already imports it — Law/Scar are not ABSENT.
9. **WorkUnit tool fields do not survive restart.** Settle: DagStore round-trip test (expected fail today).
10. **Law/Scar physical attack is still synthetic.** CAPABILITY_GRAPH `NO_INDEPENDENT_LAW_REFUTATION`. Settle: a non-synthetic result that moves a named law’s scope down.
11. **This sparse tree has zero `hcli/` files on disk.** Absence-from-disk is not absence-from-repo. Audit used `git show HEAD`.
12. **Live daemon may be running uncommitted `hcli/`.** This audit is HEAD, not that tree. Settle: read-only digest compare, no signals.

## Authority limits

- `BLOCKED_AUTHORITY`: GPU/Metal/ANE physical performance, protected benchmarks, thermal/power — not run.
- `BLOCKED_HARDWARE`: FPGA/U50, DGX, eGPU — absent.
- Uncommitted daemon `hcli/` vs HEAD: not compared.

Machine-readable companion: `receipts/audit/aud18-schema-and-api-surface.json` (every claim has `evidence` or an explicit blocker).
