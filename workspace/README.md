# Workspace layout

`workspace/` keeps Hawking's campaign material, documentation, and local
operational output out of the runtime repository root.
The root remains the ten entry points needed to build and run the product.

- `campaign/` — configuration, evidence, governance records, reports, and
  research material. Configuration is grouped by source/model; control is
  split into catalog, ledgers, verdicts, receipts, and rungs, with small
  subject buckets beneath the high-fanout groups. Odyssey follows the same
  rule: its domains, program, resources, state, and records are distinct
  browsable areas instead of a flat list of one-file folders.
- `docs/` — reference, guides, implementation notes, plans, history, and
  documentation assets.
- `ops/` — deployment descriptors, local package tooling, generated build
  output, logs, and long-running operational scripts.
- `quality/` — repository-level test fixtures.
- `vendor/` — third-party source retained in-tree and addressed by Cargo.

Historical receipt contents are deliberately not rewritten during layout moves:
their embedded paths remain provenance statements. Live code resolves the new
physical locations instead.
