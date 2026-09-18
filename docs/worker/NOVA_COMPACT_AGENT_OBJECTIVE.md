# Nova-Compact-Agent

Future Star-family specialization objective, not a training claim.

Train and evaluate a worker that preserves deep internal reasoning while
serializing only machine-useful state across the Hawking boundary:

- emit `hawking.compact_worker.v1` packets with `READ`, `PLAN`, `MUTATE`,
  `REVIEW`, `DONE`, and `BLOCKED` statuses;
- reuse source anchors and artifact references instead of reproducing source,
  digests, leases, authority, or prior evidence;
- emit compact uncertainty and next-evidence requests;
- express mutation intent as anchor, operation, replacement body, and tests;
- never restate the objective or narrate tool calls unless explicitly asked;
- distinguish internal reasoning depth from external output verbosity;
- optimize accepted-effect rate, safety, latency, and real provider cost.

Evaluation must run through the canonical Hawking acceptance funnel, including
stale-anchor rejection, no-op rejection, RED-before-GREEN, repo.edit, tests,
Git evidence, and independent review. Short output alone is not success.
