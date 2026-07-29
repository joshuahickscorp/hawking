# Temporal Gravity Numeric Parity near-tie fallback — Revision 4 insertion closure

Revise the Revision-3 candidate in place. Preserve every earlier default-off,
source-body-free, no-MOP, no-heavy-run, false-fence, immutable-lease,
complete-DAG, transactional-replacement, durable-journal, calibration, and
no-promotion requirement.

The only authorized predecessor has base
`0690d5e6ff7314cf2d26ebcf1db2cece69de2222` and these exact blobs:

- `crates/hawking-core/build.rs`:
  `e2217bd72ca68c15df0e1a2135eda15df7d602ed`;
- `crates/hawking-core/src/gravity_glm.rs`:
  `d17bd291913bec0be81928954986c1f5b53f13f6`;
- `crates/hawking-core/src/gravity_glm_resident.rs`:
  `e86f04b83a57d633605cac21b7c1557836653366`;
- `crates/hawking-core/src/numeric_parity.rs`:
  `33053ce8b70350446b0375b7eefa8c64b8cb0ec6`.

Refuse before editing if any differs. Modify only these four paths. Remove
`.serena`.

## 1. Freeze the Revision-3 insertion failure

Revision 3 reports 64 green tests, but the production insertion is still
nonfunctional:

- resident router/DSA/head call sites invoke `topk_desc_near_tie_device` or
  `topk_desc_near_tie` without the resident session;
- `topk_desc_near_tie_device_session` is imported but unused;
- host router/DSA/head call sites also use score-only non-session helpers;
- `GlmSession::with_near_tie` and an authority-aware device helper are dead
  code;
- when enabled, those wrappers pass `None` and refuse instead of executing the
  owned authority transaction.

The earlier contract explicitly named this failure. Helper-only R3 tests do not
close it. Do not count default-off historical behavior or expected refusal as a
successful enabled insertion.

## 2. One explicit run-owned session

Thread one explicit `Arc<NearTieSession>` and its opaque authority provider from
the public generation/request owner through:

- host `GravityGlm`/`GlmSession` construction, prefill, and decode;
- resident session/model construction, reset, prefill, and decode;
- every host and device router group selection, expert selection, DSA ranking,
  final-head argmax/top-k, and any downstream consumer.

Every enabled load-bearing call site invokes the session-aware entry with the
same run/request identity and a domain-specific immutable authority lease.
Remove production dependence on `near_tie_active_session`, process globals,
environment lookup, or test bridges. Score-only wrappers remain default-off
compatibility helpers only and must refuse if enabled.

Session reset/drop closes the journal writer, releases leases, invalidates
generations, and cannot leak state into a later request or concurrent model.

## 3. Real insertion authority

At each real call site, build the authority transaction from provider-owned
leased original inputs, never from the rounded score vector:

- router: activations, gate/bias roots, group mask/order, corrected and
  uncorrected score derivation, weights, and slots;
- DSA: query, key rows, head weights/scales/mask, ranked candidates, and
  attention consumer;
- head: residual, RMS weight/epsilon, head payload/codec, complete logits,
  ordered top-k, token, and next-token publication.

Bind typed byte ranges, shape/stride/offset/endian/codec, address generations,
device/executable identities, and run/request/token/layer/domain. Revalidate
before recomputation, prepare, replacement, device completion, commit, and
publication. Missing, empty, stale, caller-hash-only, cross-run, or rounded
authority refuses before mutation.

The fast scores and downstream replacement must be proven outputs/consumers of
that exact leased DAG. Preserve the corrected router mask/order and exact
transcendental policy from Revision 3.

## 4. Transactional insertion and durable terminal

For every router/DSA/head insertion, capture the actual mutable host/device
state, prepare the immutable decision receipt, apply the authoritative
replacement, wait for checked device completion where applicable, durably
commit the prepared seal, and only then publish.

Inject failure before and after every capture, apply, encode, submit, completion,
prepare, file fsync, rename, directory fsync, commit, and publication boundary.
Pre-submit failure rolls back byte-identically. Submitted failure poisons every
touched generation and requires verified reset. No partial route/descriptor,
DSA ordering/cache, head token, residual, KV, sequence, trace, or receipt may
escape.

Recovery accepts committed-only chains with exact predecessor, prepared seal,
terminal journal hash, writer/run identity, and immutable payload. Prepared
tails, mutated prepared decisions, duplicate sinks, reused run IDs, lock
replacement, torn/corrupt frames, output-unset fixtures, or fsync errors refuse.

## 5. Executable production insertion matrix

Use source-body-free host and real Metal fixtures. With near-tie enabled and
valid leases, exercise every load-bearing host/resident router, DSA, and head
call site and prove:

- the owned session call counter increments at that exact site;
- the score-only/global helper counter stays zero;
- original-input FP64 recomputation and exact replacement occur;
- the exact prepared/committed terminal is durable before output;
- concurrent runs/models never mix sessions, leases, journals, or locks.

For each site also run missing/stale/mutated/cross-run authority, zero-input
DSA, omitted bias/RMS/head roots, router mask incoherence, device failure, and
publication failure. Assert the exact refusal/rollback/poison state.

Tests must fail if any production call is switched back to the non-session
helper or if `GlmSession::with_near_tie`/resident equivalents are unused.
Remove dead session APIs and all compiler warnings in the four authorized
paths. Run Clippy with warnings denied.

## 6. Physical timing and qualification

Insertion timing comes from real monotonic spans and physical device events:
authority acquire/revalidation, FP64 recompute, prepare/fsync, replacement,
device completion when observed, commit/fsync, rollback/poison, and total.
Never derive component percentages or label a host fence as Metal D2H.

Qualification remains unavailable without a complete frozen calibration map,
true FP64 original-input authority at every wired boundary, complete coverage,
and passing p50/p95 overhead budgets. Return raw samples and coverage counts.
No fixture result is `BASE_TRUE_TPS`, TG, HIDE promotion, or capable-provider
evidence.

## 7. Exit

Return exact Git blobs/SHA-256 identities, compiler-clean output, full test
counts, per-call-site session/global counters, authority/replacement/failure
matrix, journal recovery evidence, and physical timing. Keep every product flag
default-off and all protected claims/fences false.
