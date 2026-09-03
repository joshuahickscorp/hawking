# aud14 — codebase census and topology map

Source archaeology at `04193ccbc` (`fix(hcli): a graceful stop, and a successful repair, no longer end the mission`). This lane did not implement, refactor, or rewrite the roadmap. Machine-readable twin: `receipts/audit/aud14-codebase-census.json`.

Evidence tier for every structural claim: **STATIC_VERIFICATION** (git tree, path-qualified symbol uses, `include_str`, Cargo path-deps) or **SOURCE_INSPECTION** (file existence, rustdoc, ROADMAP_STATE). **No `MEASURED` claims.** GPU/ANE/FPGA behaviour was not executed. `cargo test` / pytest were not run.

## Bounds — enumerated vs sampled

**Enumerated exhaustively:** all 15,913 git paths; workspace crate graph; hawking-core `lib.rs` 70 mods and their path-qualified callers across 949 crate `.rs` files; hawking CLI subcommands; hawking-serve and hide-serve routes; pyproject scripts; HCLI slash commands and 50 `ToolSpec("…")` names; Metal `kernel void` defs; `all_shader_sources()` membership; `load_engine` match arms; `Engine::generate` call sites in hawking / serve / bench; `ANEProvider(` and `run_fpga_preboard` via `git grep`.

**Sampled, not faked complete:** per-function/class census inside 710k Rust + 1.28M Python lines; Python call graphs in `hcli/` (not on disk here) and `tools/`; non-literal Metal `get_function(fn_name)` resolution; receipt JSON field schemas; `workspace/campaign` blobs.

This worktree is sparse: 5,755 paths on disk, 10,158 git-only. A file missing here is **not** evidence it is `ABSENT`. `hcli/` was read with `git show` / `git grep`.

---

## Counts

| class | files | lines (git grep) |
|---|---:|---:|
| git paths | 15,913 | — |
| JSON | 9,076 | mostly receipts/campaign |
| Python | 2,479 | 1,287,238 |
| Rust | 1,030 | 710,476 |
| Markdown | 583 | 130,442 |
| Metal | 58 | 50,324 |
| receipts/ tree | 4,121 | — |
| Python test files (heuristic) | 926 | — |
| Rust `crates/*/tests/*.rs` | 195 | — |

Python is the larger corpus. `tools/` alone is 1,544 files / 837,573 lines.

### Rust by crate (git loc)

| crate | loc | role |
|---|---:|---|
| hawking-core | 509,537 | inference engine + 233 examples (245,356 of those lines) + 158 tests |
| hide-backend | 54,440 | HIDE host |
| hide-kernel | 22,064 | HIDE kernel |
| hawking-context | 11,945 | HIDE-facing context |
| hide-fleet / hide-core / hawking-index / hawking-serve / hawking | 8.3k–6.2k | see topology |
| hawking-comms / hawking-eval / hawking-perception / hide-gateway | 1.2k–2.2k | **no external crate callers** |

hawking-core split on disk: **src 139 files / 237,396 loc**; **examples 233 / 245,356**; **tests 158 / 26,758**; **shaders 46 / 46,083**. The example tree is larger than the library. That is surprise S1.

### Python by tree

| tree | files | loc | tests (heuristic) |
|---|---:|---:|---:|
| tools/ | 1,544 | 837,573 | 668 |
| lab/ | 458 | 240,103 | 146 |
| hawking-experiments/ | 166 | 86,989 | 19 |
| hcli/ | 196 | 83,523 | 78 |
| ramanujan/ | 54 | 18,430 | 12 |
| civilization/ | 5 | 3,550 | 2 |

---

## Topology map

```
                     ┌─────────────────────────────────────────┐
                     │  HCLI control plane (Python)            │
                     │  hcli / jhcli / hawkingd                │
                     │  slash cmds + ToolRegistry + AgentOS    │
                     └───────────────┬─────────────────────────┘
                                     │ HTTP / spawn / receipts
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌─────────────────┐      ┌────────────────────┐      ┌────────────────────┐
│ shipping Engine │      │ organ / complete-  │      │ HIDE product graph │
│ hawking CLI     │      │ binary research    │      │ hide-* crates      │
│ hawking-serve   │      │ DSV4 modules       │      │ hawking-context/   │
│ hawking-core    │      │ qwen30/38/80       │      │ index/orch/research│
│ load_engine →   │      │ complete runtimes  │      │ hide-serve /v1/hide│
│ Engine::generate│      │ 233 examples       │      └────────────────────┘
└────────┬────────┘      │ research_server bin│
         │               └────────────────────┘
         ▼
   Metal library (36 shader files concatenated at runtime)
   GGUF / .gravity artifacts
```

Constitution as observed in source (not rewritten): **five eras**, active era **I** in `civilization/ROADMAP_STATE.json`; **three Odysseys** with acceptance receipts I/II/III; **no Era VI module**; FPGA under Accelerator/Fusion (`hcli/agentos/fpga_preboard.py`, `tools/accelerator`, `tools/future/fpga_*`); Theia is `tools/theia` (24 Python files) — one bounty package, not a civilization.

`H-ROADMAP.md` is **not in git**. `ROADMAP_STATE.json` points at `/Users/scammermike/Downloads/H-ROADMAP.md`. Surprise S3.

---

## Layer 1 — shipping inference runtime (`INTEGRATED`, static)

**Crates:** `hawking` → `hawking-core`, `hawking-serve`, `hawking-bench`, `hawking-speculate`. `hawking-serve` takes `hawking-adapters` with `default-features = false`, so the hawking binary does **not** link hide-core/hide-protocol. The Cargo.toml comment that says so is true.

**Dispatcher:** `hawking_core::model::load_engine` (`crates/hawking-core/src/model/mod.rs:80`).

Callers of `load_engine(` (the symbol, not an import):

- `crates/hawking/src/main.rs` — five sites (generate / related)
- `crates/hawking-serve/src/lib.rs:640`
- `crates/hawking-bench` competitors + prefill + decode suites
- many `crates/hawking-core/tests/*`

`Engine::generate` callers: hawking `main.rs` (4), hawking-serve `lib.rs:1025`, hawking-bench (3). That is the product decode path.

**Architectures `load_engine` actually matches:** Gravity (`.gravity` / activation-aware), mixtral-as-llama, llama/mistral, deepseek2, qwen2/qwen2.5, qwen-moe, rwkv7. Unknown arch errors out.

**Not in that match:** `qwen30_complete_runtime`, `qwen80_complete_runtime`, `qwen38_hybrid_decode`, every `gravity_deepseek_v4_*` module. Those are sibling research surfaces. Surprise S9.

**hawking CLI subcommands (enum Cmd / GravityCmd):** `serve`, `gravity {plan, condense, serve, execute, verify}`, `generate`, `tokenize`, `bench`, `autotune`, `bench-q4k-shapes`, `doctor`, `profile-rank`, `stats`, `version`, `batch-hash`, `shader-hash`, `bake-sidecar`, `press`, `fit`. Comment in `main.rs`: `bench-kernel` and `bench-server` extracted to hawking-bench; `studio` extracted to a hawking-lab pack that is **not** a workspace member.

**HTTP (hawking-serve):** `/healthz`, `/metrics`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/hawking/tokens`, `/v1/hawking/generate`, `/v1/hawking/context`, `/v1/hawking/surface`. `POST /v1/responses` and `POST /v1/messages` are wired to explicit not-implemented bodies.

Metal: sources `include_str!`'d and compiled at runtime (`MTLDevice::newLibraryWithSource`). `all_shader_sources()` concatenates **36** shader files (plus `strand_bitslice.metal` under feature `tq`). Ten files in `shaders/` are **not** in that set — they are pulled from examples. Surprise S7.

---

## Layer 2 — organ / complete-binary research (`CALLABLE` / `TESTED`, not the dispatcher)

hawking-core `lib.rs` is dominated by DeepSeek-V4 organ modules whose own docs say they are not Engine, not serve, not TPS. Path-qualified callers exist, but they are mostly `examples/` and sibling DSV4 modules.

| module | path-qualified caller files | of which examples | tests |
|---|---:|---:|---:|
| metal | 206 | 71 | 83 |
| model | 172 | 105 | 35 |
| gravity_deepseek_v4 | 69 | 37 | 3 |
| kernels | 41 | 9 | 9 |
| gravity_deepseek_v4_layer0_prefix | 35 | 19 | 0 |
| gguf / tokenizer | 35 each | 1 / 19 | 14 / 3 |
| **kernel_bench** | **0** | 0 | 0 |
| **gravity_deepseek_v4_runtime_binding** | **0** | 0 | 0 |

Two pub mods have **zero** `crate::mod::` / `hawking_core::mod::` callers outside the defining file. Imports were counted separately and also zero. That is the call-site standard, not an import count.

`qwen80_complete_runtime` callers: sibling model files, `research_server.rs`, and a stack of `ascension_qwen80_*` examples. Same pattern for qwen30 and qwen38 hybrid.

`crates/hawking-core/src/bin/{research_server,hawking-static-kernel-verify}.rs` are auto-discovered bins (no `[[bin]]` in Cargo.toml).

---

## Layer 3 — Python control plane (`INTEGRATED` as wiring; daemon not measured)

`pyproject.toml` scripts: `hcli = hcli.cli:main`, `jhcli` same, `hawkingd = hcli.hawkingd:main`.

Slash commands from `hcli/command_registry.py` `COMMANDS` (the table both `/help` and completion use):

`/help /status /models /model /tools /provider /flash-next /receipts /processes /goal /bank /ultragoal /mission /steer /grok /cancel /context /compact /clear /resume /quit /land`

Tool registry: 50 positional `ToolSpec("name", …)` registrations (aliases like `fs.read` / `filesystem.read` included). Names include `processes.list/summary/orphaned`, `odyssey.*`, `vmcp.*`, `accelerator.benchmark`, `gravity.experiment`, `grok.swarm.*`. **Registration is not invocation.** G009 already exists for reachability of those tools; this census does not re-inflate it.

`ANEProvider(` is constructed **only** in `hcli/test_ane_provider.py`. `hcli/agentos/__init__.py` imports the class. Import ≠ call. Classification: `TESTED`, not `INTEGRATED`. Hardware has a real ANE; this lane did not run it (`BLOCKED_AUTHORITY` for physical ANE facts). Surprise S11.

`run_fpga_preboard` **is** called from `hcli/agentos_cli.py` (subcommand `fpga-preboard`). FPGA board is `ABSENT` → callable software, `BLOCKED_HARDWARE` for board facts. Atlas files that *name* the path as evidence are not callers of the function.

---

## Layer 4 — HIDE (`CALLABLE`, parallel product)

hide-core ← almost every hide crate and also hawking-context/index/orch/research/events.

hide-serve HTTP: `/v1/hide/intent|events|connector|rpc|initialize` plus `/healthz`.

`hide-backend/src/bin/{haider,hcli,hide-headless}.rs` exist with no `autobins = false` — Cargo will build a **second** `hcli` binary over `hide_backend::hcli_bridge`. That is not the Python package. Surprise S8.

hawking-context, hawking-index, hawking-orch, hawking-research, hawking-events sit on the HIDE graph, not on `hawking generate`.

---

## Layer 5 — isolated scaffold crates (`SCAFFOLDED`)

Zero `hawking_comms::` / `hawking_perception::` / `hawking_eval::` / `hide_gateway::` path uses outside their own crate:

- `hawking-comms` — L1/L2/L3 sealed packet (Bible §20)
- `hawking-perception` — document pipeline stubs (Bible §19)
- `hawking-eval` — support_halo
- `hide-gateway` — retrieval/tool gateway scaffold
- `hawking-index-query` — leaf JSON-query bin, no reverse crate dep

Being a workspace **default-member** is not integration.

---

## Layer 6 — tools, lab, experiments, future

| path | what it is |
|---|---|
| tools/future (538 py) | largest Python package; name is `future` |
| tools/headless (275) | headless evidence producers |
| tools/condense (127) | condensation |
| tools/odyssey (114) | Odyssey I/II/III machinery |
| tools/accelerator (83) | Fusion/accelerator atlas, not FPGA execution |
| tools/theia (24) | one bounty model + tests |
| tools/roadmap (14) | roadmap-support parser/auditor — not H-ROADMAP.md |
| tools/haider (90 files) | historical Haider productization; architecture doc says product is `hcli/` |
| lab/operators (~269) | experiment operators (qwen30/80, glm52, dsv4f, doctor, gravity) |
| hawking-experiments | frankenstein / superwave / prometheus |
| receipts/future (1,054) | future-named receipts, not capability |
| workspace/campaign (5,555) | campaign records/evidence blobs, not source |
| ramanujan/scaffold | scaffolded research OS |

Context-pack entrypoints (`tools/llama_conditional_student_probe.py`, `nos_pipeline.py`, `gravity_share_crosslayer.py`, `wide_battery.py`, `greedy_divergence.py`) are root scripts under `tools/`. They were not executed.

---

## Metal kernel inflation trap (S14)

| metric | n | what it actually is |
|---|---:|---|
| `kernel void` defs | 822 (813 unique) | definitions in `.metal` files |
| quoted in any Rust | 793 | includes catalogs, tests, rustdoc |
| `static_kernel_name` identity arms | 409 | name catalog in `metal/mod.rs`, not dispatch |
| quoted in a dispatchish file (`get_function` / `dispatch_threadgroups` / `set_compute_pipeline`) | 557 src / 149 example-only / 107 none | still includes the catalog file |
| shader **files** in `all_shader_sources()` | 36 | the runtime compile set |

Do not report 793 dispatched product kernels. Definitions without a dispatch of **that name** are not capability.

---

## Duplication (conceptual vs mechanical)

1. **HCLI name** — Python `hcli`, Cargo `hide-backend` bin `hcli`, historical `tools/haider`. Three surfaces, one product entrypoint (`pyproject.toml`).
2. **TOKEN_NS** — unified `hawking_core::token_ns` plus per-vehicle ledgers for Q80, Q80-mixed, Qwen3.8, DSV4F. Conceptual duplication with an adapter module; not one file copied four times.
3. **strand-quant** — vendor tree excluded from workspace; optional `tq` feature ports bitslice Metal into hawking-core. Intentional absorb.
4. **Two runtime families** — GGUF `load_engine` vs complete-binary/hybrid/DSV4 organ graphs. Parallel, not aliases.
5. **Stale root `src/`** in sparse-checkout; HEAD has no workspace-root `src/`. Bins live under crates.

---

## Orphans and historical-only

| symbol | classification | why |
|---|---|---|
| `hawking_core::kernel_bench` | `SUPERSEDED` | rustdoc still says `hawking bench-kernel`; CLI extracted; zero path-qualified callers |
| `hawking_core::gravity_deepseek_v4_runtime_binding` | `SCAFFOLDED` | self-described non-runtime sidecar; zero path-qualified callers |
| hawking-comms / perception / eval / hide-gateway | `SCAFFOLDED` | no external crate callers |
| gemma2/phi3/mamba2 smoke tests | `SUPERSEDED` | still call `load_engine`; dispatcher rejects those arches; `hawking-adapters-extra` is a campaign JSON, not a crate |
| Era-VI strings under `tools/future` and `receipts/future` | `STALE_ROADMAP_TEXT` | constitution: exactly five eras |

`#[path = "…"]` files (`artifact_aap.rs`, `artifact_pq.rs`, `gravity_deepseek_v4_streamed_batched.rs`, `qwen80_hybrid_token_graph.rs`, `qwen80_multi_layer_same_runtime_encode.rs`) looked undeclared; they are submodules, not orphans.

---

## Classification snapshot (vocabulary as specified)

| thing | class | evidence |
|---|---|---|
| hawking CLI → load_engine → generate | `INTEGRATED` | call sites listed above (static) |
| hawking-serve HTTP generate | `INTEGRATED` | `lib.rs:640` load, `lib.rs:1025` generate |
| DSV4 / Qwen complete-binary organs | `CALLABLE` / some `TESTED` | examples + a few `tests/`; not dispatcher |
| HIDE stack | `CALLABLE` | crate graph + hide-serve routes; not hawking bin |
| isolated crates | `SCAFFOLDED` | zero external path-qualified crate uses |
| kernel_bench | `SUPERSEDED` | CLI gone, module remains, no callers |
| runtime_binding | `SCAFFOLDED` | no callers |
| ANEProvider | `TESTED` | constructed only in tests |
| FPGA preboard | `CALLABLE` + `BLOCKED_HARDWARE` | CLI calls software; no board |
| Theia | `TESTED` | package + tests; one bounty model |
| tools/future | `SCAFFOLDED` (tree-level) | 538 files named future; not individually classified |
| full unattended mission / TPS / EBPW | `BLOCKED_AUTHORITY` | not run; daemon not touched |
| any GPU timing | not `PHYSICALLY_MEASURED` | this lane did not dispatch a kernel |

No `END_TO_END` token generation, no `PHYSICALLY_MEASURED`, no `ADVERSARIALLY_VERIFIED` beyond noting that civilization mutation tests **exist** (not run).

---

## Surprises (loud)

1. **Examples outgrow the library** (245k vs 237k loc in hawking-core). Settle: stop citing example binaries as the Engine.
2. **Python outgrows Rust** (1.28M vs 0.71M lines). Settle: census must include tools/hcli/lab.
3. **H-ROADMAP.md is not in the repo.** Settle: whether the decree is only on disk under Downloads.
4. **kernel_bench is an advertised CLI that nothing calls.**
5. **runtime_binding sidecar has no writer/caller.**
6. **Four workspace crates are islands.**
7. **Ten shader files are example-only.**
8. **Two different `hcli` binaries.**
9. **load_engine never sees Qwen3.8/30/80 complete/hybrid runtimes.**
10. **gemma2/phi3/mamba2 tests target extracted arches whose pack is not in tree.**
11. **ANE is imported, not constructed, in production.**
12. **Five-era / three-Odyssey / Theia-as-bounty / FPGA-in-Fusion holds in topology; `tools/future` still contains Era-VI-adjacent text.**
13. **Sparse cone lists root `src/` that git does not have.**
14. **Kernel-name counts inflate unless dispatch is required.**

What would settle the physical half: run `hawking generate` on a sealed artifact (token identity), instantiate `ANEProvider` on this M3 Ultra outside tests, and a real U50 board for FPGA. This lane is not authoritative for those.

---

## What this census is not

Not a rewrite of H-ROADMAP.md. Not an implementation campaign. Not a claim that receipts prove producers. Not a claim that a passing unit test is board reality. `civilization/ROADMAP_STATE.json` percentages were **not** re-derived and are not treated as capability.

JSON: `receipts/audit/aud14-codebase-census.json` (`hawking.audit.codebase_census.v1`).
