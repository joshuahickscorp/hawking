# ingestion_fixture_v0 — **FIXTURE**, not training data

Synthetic items created by the `odyssey-data` lane to prove the ingestion
pipeline and the train/eval contamination barrier end-to-end.

- **Do not** treat this as an Odyssey training corpus.
- **Do not** report metrics as if these were real math/support examples.
- Items intentionally include exact and near-duplicate leaks against the sealed
  support-halo corpus and T0 hidden memberships so the barrier can be tested.
