# Temporal Gravity router-bias residency — Revision 2 authority closure

Revise the refused Revision-1 candidate in place. Preserve default-off
isolation, the rejected-C2 prohibition, abortable A→B transaction ownership,
checked completion, bind-once warm behavior, source-body-free fixtures,
no-wave/no-ICB constraints, and every false authorization/MOP/TG/HIDE fence.

The only authorized predecessor has base
`4e891c4ab092a569bebc4a9800ab33dd61932a28` and these exact blobs:

- `crates/hawking-core/src/gravity_glm.rs`:
  `668360d2cae3c20cfd4a5ea4d8a9ac37e1858e73`;
- `crates/hawking-core/src/gravity_glm_resident.rs`:
  `b28c6d2c4f73b0e9ce8e0668bf04db4471a3884e`;
- `crates/hawking-core/src/metal/mod.rs`:
  `93dea3d0b70e5c9c5814ab6eddcc0cc278fecbf9`;
- `crates/hawking-core/src/cost_ledger.rs`:
  `4953bdb913bd65bbc447851a59468f07d953d47e`;
- `crates/hawking-core/tests/gravity_glm_router_bias_residency_bind_once.rs`:
  `a38c49546009f8845702c776eb55322b11219040`.

Refuse before editing if any differs. Modify only those five paths. Remove
`.serena`.

## 1. Freeze the Revision-1 refusal

Revision 1 passes 15 unsandboxed Apple M3 Ultra integration tests, including
cold plus 200 warm hits and the A→B failure matrix, but remains `REVISE`:

- production `lease_for_tensor` collapses distinct same-byte allocations into
  the same lease/generation and warm hit;
- an old lease remains a warm hit after artifact-generation or
  weight-cache-generation bumps;
- source address, artifact hash, and generations are synthesized from
  constants/counters instead of the artifact-owned allocation;
- callers can emit arbitrary “physical” events and derive a complete-looking
  receipt without executing Metal;
- production observation uses a legacy unchecked command buffer, while the
  checked abortable binder is not the governing path;
- ordinary session reset bypasses verified fencing and can clear/reuse state
  without resetting poison/provider generations.

The special `mint_unrelated_same_bytes` test helper is not production authority
and may not mask these failures.

## 2. Artifact-load-owned source lease

Create the router-bias source lease only where the bound artifact tensor/cache
allocation is loaded. The immutable provider owns and binds:

- artifact/provider/build/session/run identities and live generations;
- exact tensor name/content hash, dtype, endian, shape, strides, byte
  offset/range/length, and codec;
- the real source `MTLBuffer` allocation identity, GPU address, storage mode,
  length, and allocation generation;
- weight-cache entry identity/generation and device identity;
- exact payload bytes/hash used by the router.

Do not derive identity from caller labels, payload equality, `Vec` addresses,
constant artifact hashes, mixed counters, or generation `1`. Two distinct
same-byte allocations have different leases and generations. Reuse of one
unchanged allocation is stable.

The provider—not the caller—revalidates every field at reserve, before encode,
before submit, after checked completion, before publish, at token observation,
and on every warm hit. Artifact, provider, session, cache, allocation, address,
range, tensor, or content generation changes withdraw the old record and force
a cold rebuild or typed refusal before use.

Production APIs accept the opaque provider lease, not a densified `Vec` plus
label. Test-only lease minting is unavailable to production and cannot create
production-shaped identity.

## 3. Preserve the correct transaction

Keep the Revision-1 A→B ownership that independently audited correctly:

- reserve failure before destination mutation preserves A;
- after reserve, withdraw A before any possible destination mutation;
- encode/commit/completion/publish failure leaves no reusable stale binding;
- if mutated state may have been observed, poison the generation;
- publication happens only after checked completion and final lease
  revalidation;
- rollback/reset never republishes A over B bytes.

Exercise real Metal failure before/after allocation, staging copy, encode,
submit, status completion, error read, revalidation, publish, and observation.
Use actual failing/error command buffers where the platform permits; an
injection after a successful completion is not completion-failure evidence.

## 4. Non-forgeable physical evidence

Only the actual allocation, buffer-copy encoder, command-buffer
creation/encode/commit/wait/status/error, generation-transition, publish, and
warm-hit APIs may append physical events. Make generic caller emission private
or require an unforgeable trace capability owned by those APIs.

Each event binds trace/run/request/token/device/queue/command-buffer/encoder/
buffer/address-generation identities, predecessor hash, monotonic timestamps,
byte range, completion status/error, and terminal hash. The receipt validates
the closed chain and reconciles it to the live trace capability.

An external caller that performs no Metal work must be unable to produce
allocation, staging, command, wait, completion, publish, or warm-hit evidence.
Tests must inject fabricated/reordered/duplicate/cross-trace events and prove
refusal. Counters derived from local formulas or manually assembled event
vectors are diagnostic only.

## 5. Governing production path and verified reset

Wire the checked abortable binder into the real resident router path. Mark
token state observed only after that exact bias participates in a committed
router decision whose Metal command buffer has `Completed` status and no
error. Remove legacy `commit_and_wait` as authority; default-off legacy behavior
may remain only outside the residency claim.

Production `ResidentSession::reset`, poison recovery, model/session reuse, and
public clear APIs all call one verified reset:

1. drain a checked device fence;
2. verify no in-flight reader/writer;
3. wipe bindings/destinations and physical trace ownership;
4. bump engine, provider, artifact/cache-visible, allocation, poison, and
   session generations as applicable;
5. remint the provider and require a cold bind.

No public API may clear poison or reuse a destination without that sequence.
Neighbor sessions remain healthy.

## 6. Executable matrix and speed disposition

Run unsandboxed real Metal:

- cold bind then at least 200 warm tokens with zero allocations, staging
  copies, uploads, command buffers, waits, or rebuilds attributable to bias;
- distinct same-byte allocations through the ordinary production API;
- old lease after each artifact/cache/provider/session/allocation generation
  bump;
- same allocation unchanged; tensor/range/shape/offset/content mutation;
- full A→B failure matrix, actual command failure, observation, poison, and
  verified reset;
- fabricated physical-event attempts;
- production router call-site observation and reset wiring.

Assert exact router IDs/weights and Numeric Parity near-tie fail-closed behavior.
Run randomized/interleaved cold/warm/rebuild order, Clippy with warnings denied,
and raw physical receipts. Fixture timing is never `BASE_TRUE_TPS`, TG, HIDE
promotion, C2 revival, or capable-provider evidence.

## 7. Exit

Return exact Git blobs/SHA-256 identities, full test output, authority
generation matrix, A→B state table, non-forgeable physical chain, cold/warm
receipt, and positive or negative disposition. Keep all product flags
default-off and every protected claim/fence false.
