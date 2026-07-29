# Temporal Gravity device-resident ordinary three-batch MLP

Implement a default-off, source-body-free kernel/encoder candidate that removes
host gate/up/down intermediate vectors from the ordinary dense and sparse GLM
MLP path while preserving the established three-projection semantics.

This is not expert-wave, table-wave, replay, ICB, a real-model benchmark, or a
TG promotion. Do not read model bodies, touch MOP, change production defaults,
or flip any authorization fence.

## Premise and goal

Current ordinary `batched_mlp` performs:

1. batched gate projections;
2. batched up projections;
3. host `silu(gate) * up`;
4. batched down projections;
5. host ascending expert-weight accumulation;
6. shared expert last;
7. host residual add.

The three `matvec_batch` calls return host `Vec<f32>` values and force
host/device synchronization. Preserve the exact arithmetic/order but keep
intermediate activations and combined output device-resident.

Target topology for supported fixtures:

- gate and up projections co-issued without a host-visible intermediate;
- device `silu(gate) * up`;
- down projection consumes the device activation;
- sparse weighted outputs accumulate in ascending expert ID, shared expert
  last, with the same f32 narrowing/accumulation behavior;
- final residual is encoded into the same final command boundary;
- at most two steady MLP dependency waits (gate/up then down) and no
  candidate-local per-expert waits; fewer is allowed only when physical traces
  and wall time improve.

## Hard separation from rejected paths

Use a new versioned flag and call path, default off. With it enabled and all
wave/table/replay/ICB flags disabled, actual entry probes must show zero calls
to:

- `moe_device_wave`;
- `moe_device_table_wave`;
- `ensure_expert_wave_scratch`;
- expert-table wave/replay helpers;
- `execute_replayable_graph`;
- `build_final_head_replay_graph`;
- ICB encoding or replay.

Do not relax or reuse an expert-wave parent predicate. Do not use wave scratch,
wave pipeline identities, table miss masks, or the rejected wave receipt as
authority.

## Exact arithmetic contract

Preserve Numeric Parity V2.1 and the ordinary host reference:

- identical gate/up/down tensor set, codec, dimensions, and call order;
- gate and up for the same expert/row pair;
- `silu(g) = g / (1 + exp(-g))` with the frozen device arithmetic contract;
- activation is `silu(g) * u`;
- down output narrows/accumulates exactly as specified by the existing backend;
- routed experts are processed in ascending expert ID;
- each routed output is weighted by the exact selected f32 route weight;
- shared expert contribution is added last with scale `1.0`;
- final residual update is the sole output mutation.

Dense and sparse paths share the same versioned mechanism where shapes allow.
Unsupported dtype, codec, tensor shape/layout, alias, device, or backend
refuses before mutation or selects the exact pre-token ordinary fallback with a
typed, separately counted non-qualifying reason.

Nonfinite gate/up/activation/down/weight/combined values refuse. Freeze signed
zero, subnormal/FTZ, finite-extreme, exp overflow/underflow, and narrowing
behavior. Do not loosen V2.1 bounds.

## Device-resident API and lifetime

Add the smallest explicit device-output matvec/batch API needed for supported
fixtures. It binds:

- artifact/tensor hash, codec/dtype/shape;
- source and destination GPU addresses and address generations;
- batch/expert/row order;
- buffer length/alignment/alias constraints;
- command-buffer/fence generation and ownership.

Buffers remain live until the completion fence. A ring/arena cannot overwrite
in-flight gate/up/activation/down/combined storage. Delayed-completion tests
must prove this at the actual fence boundary.

Do not stage device output through a host `Vec`, `read_f32`, mapped-buffer CPU
read, or full intermediate download. Tiny selected IDs/weights from the
ordinary router remain a separate C2 concern and are not zero-readback here.

## Transaction and poison boundary

Validate all bindings, aliasing, sizes, route order, weights, and command
ownership before residual/KV/sequence mutation.

Inject failures at allocation, bind, gate encode/commit/wait, up
encode/commit/wait, activation encode, down encode/commit/wait, accumulation,
shared add, residual encode/commit/wait, and trace/receipt publication.

- before any device/model-state mutation, restore exact pre-token state;
- after a device/model/KV mutation that cannot be proven rolled back, poison
  the session;
- poison refuses every later encode/mutation until a verified reset rebuilds
  buffers, address generations, and fences.

`seq_len`, residual, KV/DSA caches, route trace, expert execution slots, table
generations, buffer generations, and receipt state reconcile.

## Physical evidence

Count at actual APIs:

- command buffers submitted and synchronization points;
- encoders and dispatches per buffer/stage;
- GPU timestamps observed/missing;
- D2H/H2D/shared CPU reads/shared CPU writes/blit/set-bytes;
- allocations and buffer-generation rebuilds;
- gate/up/activation/down/combined/residual bytes.

Logical counters or source estimates cannot masquerade as physical evidence.
Timing runs have ledger off; a separate counter run has ledger on.

## Source-body-free fixtures and live Metal gates

Use deterministic direct-u8/native fixtures first; add existing small PQ or
activation-aware fixture coverage only when its device-output decode semantics
are already defined. Never open real model bodies.

Matrix:

- default-off exact ordinary baseline;
- dense new path;
- sparse one/multiple routed experts;
- shared expert present/absent;
- multi-layer and multi-token;
- hit/miss/stale/unsupported fallback/refusal;
- signed zero, subnormal/FTZ, finite extremes, NaN/infinity refusal;
- alias, undersize, misalignment, address-generation change;
- delayed completion and every injected failure;
- wave/table/replay/ICB flags all off with entry probes.

Assert complete V2.1 `score_pair.pass` and every continuous/discrete field for
residuals, outputs, expert/DSA/token decisions, not just greedy/top-k.

On a real local Metal device, run randomized/interleaved baseline/new modes over
enough warm iterations. Report wall and GPU p50/p95 plus physical topology.

## Kill criteria

The candidate remains unintegrated/non-promotable if any holds:

- V2.1 or exact decision failure;
- any wave/table/replay/ICB call;
- host intermediate `Vec`/readback on a claimed device-resident hit;
- more than two steady dependency waits for the MLP or more waits/CBs than the
  ordinary baseline;
- unexplained physical-byte increase;
- wall p50 or p95 regression beyond frozen noise tolerance;
- incomplete rollback/poison or in-flight overwrite;
- missing supported-codec proof.

If the device-resident path is slower on the tiny live fixture, report the
negative result and leave it default-off/rejected. Do not claim TG from fixture
latency.

## Authorized files

Change only the smallest necessary subset of:

- `crates/hawking-core/src/gravity_glm.rs`;
- `crates/hawking-core/src/gravity_glm_resident.rs`;
- `crates/hawking-core/src/metal/mod.rs`;
- `crates/hawking-core/shaders/gravity_pq.metal`;
- directly corresponding source-body-free tests/fixtures.

Do not modify model artifacts, capability/status/launch receipts, HIDE, MOP,
production defaults, or unrelated kernels.

Report exact files/hashes, supported/refused codecs, parity, before/after
physical topology and p50/p95, negative results, and unchanged false fences.
