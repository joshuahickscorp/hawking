# Temporal Gravity Numeric Parity V2.1 near-tie fallback

Implement the missing frozen/versioned near-tie canonical fallback in an
isolated worktree. This is a bounded correctness prerequisite for later kernel
promotion, not a flagship benchmark or permission to weaken parity.

## Authority and scope

Read:

- `NUMERIC_PARITY_V2.md`;
- `NUMERIC_PARITY_V2_1.md`;
- `NUMERIC_PARITY_V2_1_HARNESS.json`;
- `crates/hawking-core/src/numeric_parity.rs`;
- `crates/hawking-core/src/gravity_glm.rs`;
- `crates/hawking-core/src/gravity_glm_resident.rs`;
- corresponding GLM parity and fixture tests;
- current runtime receipts for router, DSA/top-k, final head, and ICB paths.

Inspect current source before choosing insertion points. Preserve exact artifact
semantics, same-backend determinism, and existing default-off optimization
flags. Do not read real weights, run heavy model work, touch MOP, or publish a
performance claim.

## Required behavior

Define one versioned `NearTiePolicy` with:

- explicit schema/policy version;
- decision domain and numeric dtype/backend;
- frozen finite nonnegative absolute and relative guard parameters;
- canonical comparison/order rule, including stable index tie-break;
- exact trigger formula using the relevant top decision margin;
- explicit fallback implementation identity;
- counters for comparisons, guard hits, canonical fallbacks, and decisions;
- semantic seal/config identity suitable for runtime receipts.

The policy must cover every load-bearing discrete decision that the current GLM
runtime exposes through bounded fixture execution:

- router within-group top-2;
- router group top-k;
- final expert top-k;
- DSA k/k+1;
- token argmax;
- every ordered token top-k adjacency and its k/k+1 boundary.

Device DSA is already load-bearing. Wire its guard explicitly even if it does
not share the router helper, or block device-DSA qualification. Do not claim
coverage for an unwired path.

The fast computation must expose every runner-up candidate needed to evaluate
those boundaries. A guard computed only from already-selected results is
incomplete. Acquire k+1 and internal router-boundary candidates without
changing committed decisions outside a guard hit.

The canonical fallback is the V2.1 FP64 authority. Freeze and bind its exact
input slice, accumulation/order rule, dtype conversions, implementation source
hash, and executable/build hash. Another deterministic f32 path is not
canonical evidence.

When the guard does not trigger, the existing fast decision remains
byte-for-byte and decision-for-decision unchanged. When it triggers, recompute
the smallest decisive slice through one deterministic canonical path and
require exact stable decisions. NaN, infinity, duplicate indices, invalid
top-k, empty inputs, and unsupported dtype/backend refuse.

The committed fallback decision must replace every downstream consumer before
expert dispatch, attention, sampling, trace publication, tool/stop handling, or
state persistence. Require consistent expert indices, expert weights, execution
slots, and `sample_token == head_topk_idx[0]`.

The policy must be configurable and fully receipted. Default selection must
follow the existing Numeric Parity authority; if authority does not specify a
production default or calibrated thresholds, leave promotion default-off and
report the missing calibration instead of inventing one.

Any enabled policy uses strictly positive calibrated guards. Zero thresholds
are permitted only for an explicitly disabled/test policy and cannot qualify
coverage.

Each decision receipt carries a stable decision ID, domain, layer/token,
backend/device, policy and implementation seals, observed margin and derived
threshold, trigger result, fast decision, canonical decision, committed
decision, and counter delta. A fallback may not commit if its receipt cannot be
durably emitted. Missing, stale, duplicate, or cross-run receipts invalidate
qualification.

Define counter scope, reset and merge semantics, concurrency behavior, and
exact reconciliation with decision receipts. Bind the receipt carrier schema
and output artifact into the runtime measurement receipt.

## Tests

Add deterministic source-body-free tests for:

- exact equality and stable index tie-break;
- values just below, exactly at, and just above each guard;
- positive/negative values, zeros, subnormals when supported, NaN/infinity;
- same-backend repeated determinism;
- condition-aware cross-backend perturbations around the guard;
- router/top-k and token/top-k exact decisions;
- fast-path identity outside the guard;
- counter reconciliation and policy seal stability;
- invalid/missing/stale policy versions and receipt bindings;
- no fallback when disabled and explicit proof it is disabled.

Add a bounded microbenchmark that reports fast-path overhead and guard-hit
overhead separately. Measure three modes—disabled baseline, enabled/no-hit,
and forced-hit—at the real router, DSA, and head insertion points, including
actual synchronization/readback costs. Freeze fixture sizes, warmups,
iterations, timing statistic and percentiles, backend/device/build identity,
policy seal, and machine-readable output schema. Helper-only timing is
insufficient and the benchmark must not claim whole-model TPS.

## Authorized files

Change only the smallest necessary subset of:

- `crates/hawking-core/src/numeric_parity.rs`;
- `crates/hawking-core/src/gravity_glm.rs`;
- `crates/hawking-core/src/gravity_glm_resident.rs`;
- directly corresponding tests/benchmarks.

No status, artifact, capability, launch, or authorization file may change.

## Acceptance

Run targeted and affected Rust tests, formatting/lints available in the
repository, the bounded microbenchmark, and `git diff --check`. Report exact
files/hashes, policy semantics, paths actually wired, measured microbenchmark
cost, paths still unwired, and calibration still required.

State explicitly that no capable provider, TPS/TG milestone, real model
measurement, or MOP action occurred and all fences remain false.
