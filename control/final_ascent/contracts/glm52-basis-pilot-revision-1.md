# Revision 1 — make the bounded GLM basis pilot promotion-valid

The first measured pass is useful and remains preserved as evidence, but it may
not be promoted in its current form. Reconcile these review findings in the same
pilot worktree and session.

## Finding A — requested ranks are conflated with capped ranks

The receipt includes points requested at rank 256/512 even when the actual
`total_rank` is 164 or 256. Those points currently enter the rank-256/rank-512
aggregate and floor checks. This can make a capped diagnostic masquerade as a
calibrated-rank measurement.

Required correction:

- A rank-specific aggregate or floor check may include a point only when
  `total_rank == requested_rank` and `rank_capped == false`.
- Keep capped points in per-tensor diagnostics with both requested and effective
  ranks.
- Report the included/excluded tensor counts for every aggregate.
- Add a deterministic regression test proving a capped rank-512 point cannot
  enter the rank-512 floor.

The low-traffic expert 100 has only 205 routes and is diagnostic only; it must not
be a promotion-panel minimum or median.

## Finding B — down projection is measured only in the production output space

The first pass correctly uses real routed SwiGLU rows to score `down_proj`, but it
still constructs the basis from 6,144-wide residual activations and projects the
`[6144,2048]` down matrix on its output side. The real results are decisive
negative evidence for that production representation:

- L74 e118 rank-256 cosine about 0.218;
- L38 e73 about 0.247;
- L38 shared about 0.351;
- L5 e11 about 0.358.

Do not overwrite or hide this result. Label it
`production_output_side_down_negative_control`.

Add the activation-matched comparison required to decide whether down is
salvageable:

1. derive real `Z_fit` and `Z_hold` from the same route-conditioned
   `X_fit`/`X_hold` using resident true gate/up weights;
2. build centered, uncentered, and explicit-mean bases in 2,048-wide `Z_fit`
   input space;
3. project `down_proj` on the input side and score on `Z_hold`;
4. use identical total direction count and exact byte accounting across arms;
5. call this `activation_matched_input_side_down`, and use this—not the negative
   control—as the down promotion metric.

Add tests proving fit/holdout correspondence, the 2,048-wide input basis,
no Gaussian path, equal bytes, and separation of the two down analyses.

## Finding C — promotion panel and verdict semantics

- Publish separate aggregates for:
  - promotion-grade high-traffic routed gate/up/down at early/middle/late layers;
  - shared MLP;
  - attention/router controls;
  - low-traffic diagnostics.
- Floors must be evaluated on the promotion-grade panel, while per-organ failures
  remain visible.
- `beats_null`, reconstruction error, and the invalid all-row diagnostic remain
  non-promotional.
- Uncentered and explicit-mean are numerically tied in the first pass. Do not
  declare explicit-mean uniquely superior when their median difference is below
  a stated numerical-equivalence tolerance. The valid conclusion is that
  retaining the mean helps centered residual; the implementation choice remains
  unresolved when B≈C.
- The verdict must contain an explicit `full_traversal_authorized: false` unless
  every preregistered bounded promotion criterion truly passes.

## Resource and receipt requirements

- Reuse the already verified five source hashes; do not fetch anything.
- Check current free disk before rerun and write only small code/receipt files.
- Preserve the first receipt either in Git history or as a clearly named
  `revision_0` evidence block/hash in the revised receipt.
- Report panel-total and per-tensor exact encoded bytes/BPW with the accounting
  scope stated. Do not call an arithmetic estimate a physical file measurement.
- Update Markdown and JSON, rerun unit tests and the real pilot, and commit all
  intended files excluding `.serena`.
- Keep all three fences false and do not change production defaults.

Return exact before/after findings, test results, measured input-side down
cosines, corrected floor checks, remaining uncertainty, and the next safe action.
