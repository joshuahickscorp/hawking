# TG K4 — descriptor-indexed non-wave three-batch MoE

## Status and dependencies

This is a source-body-free, default-off implementation contract. Do not begin
implementation until both predecessors are frozen and independently accepted:

1. the explicit run-owned near-tie FP64 authority transaction; and
2. the abortable device-resident three-batch MLP with green live-Metal parity
   and nonregressing p50/p95.

K4 removes selected expert ID/weight D2H on a warm hit. It is a new mechanism,
flag, version, and receipt. It does not revive the rejected
`c2_device_router_rev1_rejected_20260728`, expert wave, table-wave, ICB, or
replay.

Remain source-body-free. No MOP, real model, capable-provider, product TPS/TG,
HIDE promotion, or authorization transition.

## Mechanism

Use canonical device buffers as the only warm-hit selection inputs:

- score-ranked `expert_idx[k]`;
- score-aligned `expert_w[k]`;
- ascending-ID `expert_exec_slots[k]`.

Address gate/up/down tensors through the existing immutable
`DeviceExpertTensorRef`/`DeviceExpertTriplet` ABI. Freeze Rust and Metal
size/alignment assertions. A ready triplet binds exact tensor/artifact hashes,
codec, shape, buffer addresses, device, lease generation, weight-cache
generation, session/poison epoch, and completion fence.

The warm-hit topology is the accepted ordinary device three-batch predecessor:

1. all selected gate/up projections;
2. device SiLU×up;
3. all selected down projections;
4. ascending expert-ID weighted accumulation using score-aligned weights;
5. shared expert last at scale `1.0`;
6. the sole residual mutation.

Do not call:

- `moe_device_wave`;
- `moe_device_table_wave`;
- `ensure_expert_wave_scratch`;
- any expert/final-head ICB or replay helper;
- `gpu_expert_table_hit_enabled` as a parent predicate.

K4 scratch is owned by the accepted three-batch arena or a new explicitly
non-wave arena. It is never wave scratch.

## Hit, miss, and fallback

Validate descriptor readiness, kinds, shapes, generations, aliases, and device
ownership before residual mutation.

Warm hit:

- zero selected-ID D2H;
- zero selected-weight D2H;
- zero descriptor upload;
- exact V2.1 decisions and continuous values;
- descriptor/source/scratch leases retained through successful completion.

Miss, stale generation, or unsupported codec:

- guarded kernels leave residual unchanged;
- a typed miss mask may download;
- selected IDs/weights may download only after the miss is established;
- fallback uses the ordinary non-wave path or accepted host-named
  device-three-batch predecessor;
- no partial trace, dispatch, receipt, or residual escapes.

Activation-aware and unknown codecs start as typed nonqualifying fallbacks.

## Near-tie authority

Before K4 validation/dispatch, the accepted near-tie session either:

- atomically replaces complete IDs, weights, slots, traces, and device buffers
  from original-input FP64 authority; or
- refuses K4 qualification.

Host-f32 widening, peer GPU agreement, exact-tie-only fixtures, and the
rejected C2 result are not authority.

## Transaction and poison

Use the predecessor's abortable owner:

`Prepared → Encoding → Committed → CompletedSuccess|CompletedFailure`

or `AbortedBeforeCommit`.

Drop before commit never commits. Preflight every descriptor, arena, command,
authority, and fallback resource before canonical mutation. A post-submit
failure poisons the real resident session. Verified reset waits fences, clears
residual/KV/DSA/sequence/router/descriptor/arena/receipt state as required,
bumps epochs/generations, and refuses if incomplete.

## Physical evidence

Record ordered raw events at real APIs for allocations, descriptor uploads,
mapped access, blits, command creation/encode/commit/completion/failure,
waits, dispatches, generation changes, miss-mask D2H, selected-ID/weight D2H,
residual mutation, and fallback.

Required warm-hit evidence:

- `selected_id_d2h_bytes == 0`;
- `selected_weight_d2h_bytes == 0`;
- `descriptor_upload_bytes == 0`;
- `k4_hit > 0`, `k4_fallback == 0`;
- all wave/table/ICB/replay probes `0`.

Caller-only counters are diagnostics, not proof. Separate ledger-off timing
from ledger-on topology.

## Source-body-free matrix

Test direct-u8 PQ and native BF16 with:

- default-off bit/decision identity;
- `k ∈ {1,2,4,8}`;
- score order distinct from ascending execution order;
- shared present/absent;
- multi-layer/multi-token lease reuse;
- cold descriptor build and warm zero-upload hit;
- one selected triplet missing;
- stale table/weight/session/poison/device generation;
- unsupported codec/kind/bits;
- FP64 near-tie reversal or authority-unavailable refusal;
- signed zero, subnormal/FTZ, finite extremes, NaN/infinity refusal;
- alias, undersize, misalignment;
- every abort/commit/completion/reset failure;
- no residual/trace publication on miss or failure.

## Live Metal gate

Modes:

- ordinary baseline;
- accepted device three-batch predecessor;
- K4.

Use the same device/context/executable/fixture/prompts, randomized paired
interleaving, at least 20 warmups and 200 measured iterations, nearest-rank
p50/p95, ledger-off timing, and ledger-on topology.

Reject K4 if:

- complete V2.1 fails or any discrete decision differs;
- selected ID/weight D2H occurs on hit;
- a forbidden probe fires;
- waits/command buffers exceed the replaced ordinary work;
- p50 or p95 regresses beyond the frozen paired tolerance;
- a miss/failure mutates residual or publishes state;
- completion/poison/reset evidence is incomplete.

Fewer waits with worse wall time is a negative receipt.

## Authorized implementation scope

After predecessor blobs are frozen, authorize only:

- `crates/hawking-core/src/gravity_glm.rs`;
- `crates/hawking-core/src/gravity_glm_resident.rs`;
- `crates/hawking-core/src/metal/mod.rs` if the accepted abortable API needs
  additive binding support;
- `crates/hawking-core/shaders/gravity_pq.metal` only for a strictly
  non-wave batching gap;
- `crates/hawking-core/src/cost_ledger.rs`;
- `crates/hawking-core/src/numeric_parity.rs` only for coordinated accepted
  near-tie insertion;
- one focused integration test.

The implementation addendum must freeze exact predecessor blobs before edits.
Return exact blobs/SHA-256, test counts, physical counters, p50/p95, and one
disposition: `ACCEPTED_FIXTURE_ONLY`, `REJECTED_PARITY`,
`REJECTED_PHYSICS`, or `AUTHORITY_UNAVAILABLE`.
