# hawking-experiments

Archive of legacy and paused experiment campaigns. Anything not absolutely
near-current lives here — it stays searchable and can be resumed ("cooked again")
later. Active work stays in the repo root (crates, tools, receipts/headless, the
current campaign under workspace/campaign).

Populated by the consolidation pass; see receipts/headless/CONSOLIDATION.json
and receipts/headless/CONSOLIDATION2.json.

## Layout (CONSOLIDATE-2)

- `frankenstein/operators/` — live frankenstein Python operators (moved from lab/operators, files named frankenstein_*.py)
- `frankenstein/condense/` — CLI wrappers + tests (moved from tools/condense, files named frankenstein_* / test_frankenstein_*)
- `frankenstein/data/` — campaign evidence (moved from workspace/campaign/evidence/models, frankenstein tree)
- `prometheus/tools/` — prometheus package (moved from tools, prometheus tree)
- `prometheus/config/` — allocation profiles (moved from workspace/campaign/config/profiles, prometheus tree)
- `prometheus/evidence/` — research evidence (moved from workspace/campaign/evidence/research, prometheus tree)
- `superwave/` — dead Superwave data (moved from workspace, superwave tree)

`hawking-experiments` is not a valid Python package name (dash). Import the
moved modules by bare name (`import frankenstein_ablation`, `import prometheus`)
after `lab.layout.ensure_experiment_imports()` (also invoked from
`lab.operators` import and repo-root `conftest.py`). Ramanujan stays at
repo-root `ramanujan/` — it is a live package.

Live readers of historic path strings should use `lab.layout.resolve_workspace_path`.
