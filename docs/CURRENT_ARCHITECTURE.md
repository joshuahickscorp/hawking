<!-- DOC_STATUS: CURRENT -->
# CURRENT ARCHITECTURE

Checked against `receipts/headless/CODE_GRAPH.json` (schema `hawking.headless.code_graph.v1`, graph `git_head` `cba2bb657`) and the HEAD tree `fef695d26`. Where this document and the graph disagree, the graph wins for the snapshot it recorded; the disagreement is in `receipts/headless/ARCHITECTURE_CANON.json`. This file describes what **is** at HEAD, not what the graph still names.

## Control plane

One installed package: **`hcli`**, physical path **`hcli/`**, `pyproject.toml` name `hcli`, console scripts `hcli` and `jhcli` → `hcli.cli:main`. `python3 -m hcli` is the product entry. `import tools.haider.hcli` raises `ImportError` -- there is no second package and no `tools/haider` at all. Incident F24 (one file, two dotted names, two class objects) is locked by `hcli/tests/test_module_identity.py`, which keeps the fossil dotted name on purpose so a mechanical rename cannot delete the thing it guards against.

Tests live at `hcli/tests/`. The bootstrap-era fossils (`tools/hcli/bootstrap/`) are a dated record, disconnected from the control plane. `lab/hcli/` is a different product.

The graph still inventories `tools/haider/hcli/*.py` as `hcli_product` (33 files) -- a stale path; the graph has not been regenerated since the move. Those 33 basenames exist at `hcli/` on HEAD. The graph does not list `hcli/paths.py`, `hcli/persist.py`, or the ownership packages `hcli.agentos`, `hcli.doctor`, `hcli.gravity`, `hcli.vmcp`, `hcli.genomes`.

## Ownership (importable, not comments)

| Plane | Means | Where a program finds it |
|---|---|---|
| **HCLI** | Command surface, UI, headless entry, status rendering | `hcli.{cli,app,commands,tui,controller,events,workspace,grok_bridge,report_compiler}` |
| **AgentOS** | Goal, WorkUnit DAG, scheduler, repair, mutation, verifier orchestration, steering | `hcli.agentos` re-exports the same class objects as `hcli.{goal,workunit,scheduler,mission,ledger,steering,mutation,verifier_pipeline,dag_store,executors,resources,index}`. Files are not nested under `hcli/agentos/` (that would recreate F24). |
| **Runtime** | Execution, sessions, context, backends | `hcli.{runtime,engine,backends,session,context,context_budget,models,machine,max_policy,config}`. `hcli/runtime.py` is the module; a `hcli/runtime/` package would shadow it. |
| **Doctor** | Measures and prescribes | Ownership marker `hcli.doctor`; instruments stay `tools/doctor_seal.py`, `tools/gravity_doctor_*.py` (`nos_pipeline` stage 3). |
| **Gravity** | Search and compile | Ownership marker `hcli.gravity`; product stays `tools/gravity_*.py` plus hawking crates. |
| **VMCP** | Sensory evidence | Ownership marker `hcli.vmcp`; product is the `visionmcp/` package. |
| **Genomes** | Learned science | `hcli.genomes` re-exports `hcli.machine.MachineGenome` (same class object). |

## Rest of the tree (not the control plane)

- **`hawking-*` crates** — Metal inference engine, GGUF/Gravity loaders, serve/bench/speculate. Default `cargo build` surface.
- **`hide-*` crates** — HIDE agent IDE. Workspace members, not default-members. `hide-backend` talks to `hawking serve` over HTTP.
- **`lab/`** — campaign governance (`lab.rules`, `lab.receipts`, `lab.science_registry`). Out of the code-graph census.
- **`tools/headless/`** — harnesses and headless tests. Census root of the graph, not a product import of `hcli`.
- **`workspace/docs/reference/ARCHITECTURE.md`** — HISTORICAL (Rust three-layer hawking binary). Outside `docs/`; not rewritten.

## Graph vs HEAD (graph wins for its snapshot)

Graph `git_head` `cba2bb657` is an ancestor of HEAD `fef695d26`. G026 (`a76efc875`) landed between them.

| Graph fact | HEAD fact |
|---|---|
| Census roots were `tools/haider`, `tools/headless` | Now `hcli`, `tools/headless` |
| Finding `dual_import_identity` severity high | Fossil name raises `ImportError` |
| Import SCC `dag_store → max_policy → resources → workunit` | Broken by extracting `hcli.persist.atomic_write_json` |
| 88-class `sys.path.insert` coupling | Fossil inserts deleted; remaining inserts are foreign packages / harnesses |
| Product entrypoints once included `tools/haider/hcli/__main__.py` | Installed `hcli.__main__` / `hcli.cli:main` |
| `hcli.ledger` ↔ `hcli.steering` import cycle | Still present; import-time safe (`TYPE_CHECKING` / function-level) |

The graph is not rewritten. Successor evidence: `receipts/headless/NAMESPACE_MIGRATION.json`.

## Production truth that still scrapes terminal prose

`hcli.grok_bridge.parse_grok_status` parses `grok-run status --id` **human text** (`status: running (exit -)`). `GrokBridge.status()` and `extract_task_id()` take task identity and terminal state from stdout. Occupancy in `hcli.max_policy` is a process check, not this parser; `_scan_throttle` still reads task `stdout`/`stderr` for 429 text.

## Laws (machine-readable homes)

Each law is resolved by a program loading a JSON receipt and/or importing a symbol. English is not the resolver. Full map: `receipts/headless/ARCHITECTURE_CANON.json` → `laws`.

1. **tools know** — disk JSON is knowledge. Load `receipts/headless/MACHINE_GENOME.json` (`schema` `hawking.headless.machine_genome.v1`) or `hcli.machine.MachineGenome`. Do not scrape TUI text.
2. **evidence promotes** — `hcli.verifier_pipeline.command_is_admissible`; `lab.rules.apply_governance`; receipt `AGENTOS_VERIFIER_AUTHORITY.json` `attacks_closed.self_promotion_from_prose`.
3. **adversary attacks** — `python3 tools/headless/gate_adversary.py` → `NOETIC_GATE_ADVERSARY.json` (default REFUTED).
4. **no-op proves causality** — `hcli.mutation` raises `NO_OP_MUTATION`; Doctor seal refuses without a control watched to fail; live conventional control is `CONVENTIONAL_CONTROL_SET.json`.
5. **atomic truth** — `hcli.persist.atomic_write_json` (temp + `os.replace`); `MutationLock.acquire` via `os.link` of a fully-written temp.
6. **candidate preservation** — `DIRTY_TREE_PRESERVATION.json`; `CODE_ENTROPY.json` `never_delete.receipts/`; `NOETIC_NEGATIVE_SCIENCE.json`.
7. **content-fresh evidence** — `hcli.goal.assert_evidence_fresh` (size + `mtime_ns` + sha256); stale → `StaleEvidenceError`.
8. **single writer** — `hcli.resources.MutationLock`; receipt `AGENTOS_SINGLE_WRITER.json` `result`.
9. **focused context** — `hcli.goal.compile_worker_context`; workers get `ROOT_REF`, not the inlined root goal. Receipt `HCLI_WORKUNIT_FOCUSED_CONTEXT.json`.
10. **MAX equilibrium** — `hcli.max_policy.load_equilibrium` / `.hcli/max-equilibrium.json`; measured rungs in `GROK_MAX_EQUILIBRIUM.json` (`useful_equilibrium`) and `QWEN_MAX_EQUILIBRIUM.json`. Occupancy = live `grok-run` processes.
11. **historical science preservation** — `CODE_ENTROPY.json` `never_delete`; nomenclature `vestigial_means` (vestigial ≠ reclaimable); negative-science register.
12. **Noetic ontology** — `NOETIC_ARCHAEOLOGY_INDEX.json` `classification_key` {RAN, CODE_ONLY, REFUTED, PROSE_ONLY}; surviving brands in `NOMENCLATURE_CENSUS.json`.
13. **native operators** — `NOETIC_NATIVE_OPERATOR.json`: NATIVE vs ORACLE by `peak_temporary_materialization` vs parent-tensor shape.
14. **no hidden information accounting** — `NOETIC_INFORMATION_ACCOUNTING.json` seven buckets + canary `completeness`.
15. **routes as physical cost** — `NOETIC_ROUTE_LEDGER.json`; every route count is paired with parent-weight equivalents.
16. **function-space fitting** — `FRACTIONAL_BIT_CANON.json` `survival_rule`; score `Y=X@W.T` on real X (`tools/gravity_function_space_rank.py`), not weight cosine.
