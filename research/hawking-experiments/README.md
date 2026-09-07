# hawking-experiments

Archive of legacy and paused experiment campaigns. Anything not absolutely
near-current lives here — it stays searchable and can be resumed ("cooked again")
later. Active work stays in the repo root (crates, tools, receipts/headless, the
current campaign under workspace/campaign).

Populated by the consolidation pass; see receipts/headless/CONSOLIDATION.json
and receipts/headless/CONSOLIDATION2.json.

## Layout (CONSOLIDATE-2)

- `frankenstein/operators/` — live frankenstein Python operators (moved from lab/operators, files named frankenstein_*.py)
- Frankenstein wrapper/test duplicates were retired; live tooling remains under `tools/condense/` and operator source under `frankenstein/operators/`.
- `frankenstein/data/` — campaign evidence (moved from workspace/campaign/evidence/models, frankenstein tree)
- `prometheus/config/` — allocation profiles (moved from workspace/campaign/config/profiles, prometheus tree)
- `prometheus/evidence/` — research evidence (moved from workspace/campaign/evidence/research, prometheus tree)
- `superwave/` — dead Superwave data (moved from workspace, superwave tree)

`hawking-experiments` is not a valid Python package name (dash). The retained
Frankenstein operators are imported by bare name after
`lab.layout.ensure_experiment_imports()`. Prometheus is evidence/config only;
its executable package was retired. Ramanujan is retained
under `research/ramanujan/` as evidence only; live verification code is under
`tools/verify/`.

Live readers of historic path strings should use `lab.layout.resolve_workspace_path`.
