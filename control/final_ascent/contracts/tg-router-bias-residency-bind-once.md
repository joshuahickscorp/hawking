# Temporal Gravity router-bias residency — bind-once candidate

## Status and authority

This contract authorizes a source-body-free, default-off kernel candidate only.
The implementation base is main commit
`4e891c4ab0929a9876f01edf8913b9fe481c067d`.

The candidate prepares a future non-wave device router by removing repeated
warm-token router-bias uploads. It does not revive or promote the rejected
Revision-1 device-router implementation, expert wave, an ICB path, or a
table-wave path.

Authorized implementation files:

- `crates/hawking-core/src/gravity_glm.rs`;
- `crates/hawking-core/src/gravity_glm_resident.rs`;
- one new focused test under `crates/hawking-core/tests/`.

No model body, rare-route shard, MOP file, production receipt, or governing
authorization fence may be read or changed. `.serena` files are never
deliverables.

## Required mechanism

Add an explicit versioned, default-off router-bias residency mode. A session
may reuse a device bias binding only when one immutable binding record matches
all of:

- session and poison epoch;
- layer and router identity;
- artifact/tensor content identity;
- source address/buffer generation;
- destination Metal device/registry identity and buffer generation;
- dtype, endian, logical length, byte length, and layout/version;
- owning command/fence generation proving the upload completed.

A cold binding or valid generation rebuild performs exactly one bias upload.
Every later warm token with the same complete identity performs zero bias
uploads and zero bias allocation. A changed source, layer, device, destination
generation, session epoch, or poison epoch must rebuild before router encoding.

Never infer identity from pointer equality, caller labels, length alone, or
equal bytes found in an unrelated allocation. Detect source/destination
aliasing and refuse before encoding or publishing a router decision. A failed,
partial, or uncompleted upload must not become reusable.

The candidate must remain usable with expert wave, ICB, and table-wave forced
off. It may expose a fixture-only direct device-router insertion point, but it
must not depend on or relabel the rejected
`c2_device_router_rev1_rejected_20260728` path as accepted. Production and
combined hot-path resolution remains baseline-only.

## Transaction and failure rules

Binding creation is transactional:

1. reserve a new destination generation;
2. encode/upload the complete bias;
3. commit and obtain real completion evidence;
4. publish the immutable binding record.

Failures at reserve, encode, commit, completion, or publication discard the
new generation and poison the owning session if prior token state could be
observed. Poison is checked before later encode or mutation. No destructor may
silently commit unfinished work.

Physical evidence must come from the existing real transfer/command-buffer
hooks. Caller-declared counters alone are not evidence. Ledger-off timing and
ledger-on topology runs are separate.

## Source-body-free test matrix

Use deterministic tiny direct-u8/f32 fixtures and exercise:

- default-off behavior is byte-for-byte and decision-for-decision unchanged;
- cold token: exactly one bias upload;
- at least 200 warm tokens: zero further bias uploads/allocations;
- layer switch, source-generation change, destination-generation change,
  device change, session reset, and poison reset each force one rebuild;
- same-length changed bytes and unrelated same-byte allocation cannot reuse;
- source/destination alias refuses before dispatch;
- injected failure at every transaction step leaves no reusable binding;
- wave, ICB, and table-wave forced off throughout;
- exact router IDs, weights, ordering, and continuous values match a valid
  authority fixture, while near-tie authority remains fail-closed;
- separate topology evidence reports uploads, bytes, allocations, command
  buffers, waits, and dispatches from physical hooks.

Run randomized/interleaved cold/warm/rebuild modes. Any wall-time result is a
fixture insertion-point measurement only, never `BASE_TRUE_TPS` or a TG
milestone.

## Exit criteria

Return exact Git blob and SHA-256 identities for every authorized file, the
commands and complete test counts, and a clear list of refused claims.

Remain false/default-off:

- `RAMANUJAN_RESEARCH_AUTHORIZED`;
- `HIDE_KERNEL_TURN`;
- `ODYSSEY_LAUNCH_AUTHORIZED`;
- full parent traversal;
- every product TPS/TG/HIDE promotion claim.
