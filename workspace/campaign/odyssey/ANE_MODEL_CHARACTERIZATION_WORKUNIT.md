# ANE per-model Odyssey work unit

Status: `ACTIVE_SECONDARY_ODYSSEY_LANE`

This is a continuation of the existing ANE/PhysicalGraph work. It does not
create a new science, ontology, or promotion gate. The durable registry is
`ANE_MODEL_CHARACTERIZATION.json`; `ANEProvider` and `PhysicalGraph` remain
the implementation owners.

## Objective

Characterize the Neural Engine for each strategically important model and organ
in the same way Hawking characterizes CPU, GPU, and unified memory. Fold every
reusable result into Gravity's provider-neutral physical lowering and the
shared MegaKernel execution spine.

The goal is not to force every model onto ANE. The goal is to let measured
complete useful work choose the physical placement.

## Current truth

- The machine publicly reports CPU, GPU, and Neural Engine devices.
- Existing ANE evidence is static/public-API evidence only.
- No Flash-specific graph has earned ANE placement, latency, throughput, energy,
  capability, or complete-token evidence.
- The active system Python cannot import `coremltools`; the active developer
  directory is CommandLineTools rather than a full Xcode developer directory.
- The public Core ML fixture control was freshly revalidated in
  `receipts/headless/ANE_PUBLIC_FIXTURE_REVALIDATION_20260911.json`: the ANE,
  GPU, and CPU are visible; the single `ios16.mul` fixture operation supports
  all three but is CPU-preferred in the observed `MLComputePlan`. Its
  48,995--108,957 ns prediction samples are fixture-only and cannot transfer
  to Flash, KIMI, or a placement/promotion claim.
- `ANEProvider` is evidence-only and remains fail-closed.
- Metal remains the primary Flash route.

These facts stay attached to every model row. A plan or device list is never
reported as execution.

## Per-model protocol

For each model, seal the source identity and architecture anatomy first. Then
select a small organ set that represents the model's actual execution:

1. Record operation, shape, dtype, representation, state, route, and model
   source seal.
2. Run the public Core ML/MLComputePlan path when authoring and compilation are
   available. Record supported devices and preferred placement as plan facts.
3. Match CPU, GPU/Metal, ANE, CPU+ANE, CPU+GPU, and justified concurrent
   controls on the same model seal, organ, shape, and representation. The
   generated backend matrix begins UNMEASURED for CPU/GPU and carries only
   public-API plan facts for ANE; it is a work card, never a borrowed result.
   Keep all timings in integer nanoseconds.
4. Bill transfer, conversion, synchronization, residency, launch, fallback,
   and setup costs. Keep prefill, decode, organ, and complete-token walls
   separate.
5. Require source parity and capability evidence before treating an organ as
   useful. Require a protected complete-token wall before backend promotion.
6. Carry positive or negative evidence forward only under its exact model,
   organ, shape, representation, and machine scope.

## MegaKernel construction

The MegaKernel is the shared execution spine, not a giant unverified fused
kernel. It fixes the model-aware stages:

`LOAD -> DECODE -> ROUTE -> PROJECT -> ACCUMULATE -> STATE_UPDATE -> SAMPLE`

Each backend supplies a native lowering for the stages it can execute. The
model supplies organ shape, representation, route, and state layout. Fusion is
allowed only after source parity and complete useful-work measurement. ANE is a
candidate backend slot; it cannot displace the current Metal route from a
nominal device list or a plan-only row.

The state layout is now a first-class MegaKernel contract: route cache,
recurrent state, and KV state must remain model-scoped resources across
accepted decode steps. Replaying a prefix into a new cache does not count as a
decode continuation. Exact bytes, allocation, residency, and backend transfer
remain per-model measurements rather than inferred metadata.

The Rust registry also records exact existing native lowerings. Its current
Qwen-3B-shaped Metal entry is a pass-through POC and explicitly cannot be
selected or promoted. It is retained as an engineering target, not represented
as Flash-Next execution.

This makes the same physical concepts reusable across KIMI, Flash-Next, and
future Odyssey specimens while keeping model-specific evidence separate.

## Active model queue

- `KIMI_P0_OPERATIONAL`: bounded local worker/control; use for model-scoped
  characterization when it does not interfere with real worker duties.
- `Qwen--Qwen3.8-Flash-Next@34567a4712bc`: primary Pulsar candidate; current
  static work continues while the valid runtime/activation path is absent.
- Future Odyssey models enter only with exact source identity and a small
  organ census; no generic ANE claim transfers automatically.

## Next bounded actions

- Keep the public ANE probe and toolchain blocker observable.
- Populate model-specific organ rows from existing anatomy receipts.
- When a lawful compile path exists, run the smallest model-scoped plan probe.
- Lower repeated CPU/GPU/ANE stage contracts into native backends only after a
  reusable parity target exists.
- Keep Flash static Deep Gravity, resident KIMI work, and Rust Gravity
  repatriation active in parallel.

## Acceptance

This work unit is successful when it produces either a protected,
model-specific useful ANE result that improves complete work or a reproducible
model-specific blocker/ceiling that Gravity can use for placement. It is not
successful merely because ANE appears in a graph or because a compiler lists a
device.
